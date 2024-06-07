#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "stardist3d_impl.h"

// dist.shape = (n_polys, n_rays)
// points.shape = (n_polys, 3)
// verts.shape = (n_rays, 3)
// faces.shape = (n_faces, 3)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

__device__ float compute_iou_device(const float *d1, const float *d2, const float *p1, const float *p2, const float *verts, const int *faces, int n_rays, int n_faces, int use_bbox, int use_kdtree) {
    // Implement the logic for calculating IoU between polyhedra d1 and d2
    // Placeholder: return a dummy value
    return 0.0f;
}

__global__ void cuda_non_max_suppression_kernel(const float *dist, const float *points, const float *verts, const int *faces, const float *scores, bool *result, int n_polys, int n_rays, int n_faces, float threshold, int use_bbox, int use_kdtree, int verbose) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_polys) return;

    extern __shared__ float shared_mem[];

    float *shared_dist = shared_mem;
    float *shared_points = shared_mem + n_rays * n_polys;
    float *shared_verts = shared_mem + n_rays * n_polys + 3 * n_polys;
    int *shared_faces = (int *)(shared_mem + n_rays * n_polys + 3 * n_polys + 3 * n_rays);
    float *shared_scores = (float *)(shared_faces + 3 * n_faces);

    if (threadIdx.x < n_rays * n_polys) {
        shared_dist[threadIdx.x] = dist[threadIdx.x];
    }
    if (threadIdx.x < 3 * n_polys) {
        shared_points[threadIdx.x] = points[threadIdx.x];
    }
    if (threadIdx.x < 3 * n_rays) {
        shared_verts[threadIdx.x] = verts[threadIdx.x];
    }
    if (threadIdx.x < 3 * n_faces) {
        shared_faces[threadIdx.x] = faces[threadIdx.x];
    }
    if (threadIdx.x < n_polys) {
        shared_scores[threadIdx.x] = scores[threadIdx.x];
    }

    __syncthreads();

    result[idx] = true;
    for (int j = 0; j < idx; ++j) {
        if (shared_scores[j] > shared_scores[idx]) {
            float iou = compute_iou_device(&shared_dist[idx * n_rays], &shared_dist[j * n_rays], &shared_points[idx * 3], &shared_points[j * 3], shared_verts, shared_faces, n_rays, n_faces, use_bbox, use_kdtree);
            if (iou > threshold) {
                result[idx] = false;
                break;
            }
        }
    }
}

void cpu_non_max_suppression(const float *dist, const float *points, const float *verts, const int *faces, const float *scores, bool *result, int n_polys, int n_rays, int n_faces, float threshold, int use_bbox, int use_kdtree, int verbose) {
    #pragma omp parallel for
    for (int i = 0; i < n_polys; i++) {
        result[i] = true;
        for (int j = 0; j < i; ++j) {
            if (scores[j] > scores[i]) {
                float iou = compute_iou_device(&dist[i * n_rays], &dist[j * n_rays], &points[i * 3], &points[j * 3], verts, faces, n_rays, n_faces, use_bbox, use_kdtree);
                if (iou > threshold) {
                    result[i] = false;
                    break;
                }
            }
        }
    }
}

static PyObject* c_non_max_suppression_inds(PyObject *self, PyObject *args) {
    PyArrayObject *arr_dist=NULL, *arr_points=NULL, *arr_verts=NULL, *arr_faces=NULL, *arr_scores=NULL;
    PyArrayObject *arr_result=NULL;

    float threshold = 0;
    int use_bbox;
    int use_kdtree;
    int verbose;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiif",
                          &PyArray_Type, &arr_dist,
                          &PyArray_Type, &arr_points,
                          &PyArray_Type, &arr_verts,
                          &PyArray_Type, &arr_faces,
                          &PyArray_Type, &arr_scores,
                          &use_bbox, &use_kdtree,
                          &verbose,
                          &threshold))
        return NULL;

    const int n_polys = PyArray_DIMS(arr_dist)[0];
    const int n_rays = PyArray_DIMS(arr_dist)[1];
    const int n_faces = PyArray_DIMS(arr_faces)[0];

    const float * const dist = (float*) PyArray_DATA(arr_dist);
    const float * const points = (float*) PyArray_DATA(arr_points);
    const float * const verts = (float*) PyArray_DATA(arr_verts);
    const int * const faces = (int*) PyArray_DATA(arr_faces);
    const float * const scores = (float*) PyArray_DATA(arr_scores);

    npy_intp dims_result[1];
    dims_result[0] = n_polys;
    arr_result = (PyArrayObject*)PyArray_SimpleNew(1, dims_result, NPY_BOOL);

    bool *result = (bool*) PyArray_DATA(arr_result);

#ifdef __CUDACC__
    // GPU parallelism
    float *d_dist, *d_points, *d_verts, *d_scores;
    int *d_faces;
    bool *d_result;

    cudaMalloc((void**)&d_dist, n_polys * n_rays * sizeof(float));
    cudaMalloc((void**)&d_points, n_polys * 3 * sizeof(float));
    cudaMalloc((void**)&d_verts, n_rays * 3 * sizeof(float));
    cudaMalloc((void**)&d_faces, n_faces * 3 * sizeof(int));
    cudaMalloc((void**)&d_scores, n_polys * sizeof(float));
    cudaMalloc((void**)&d_result, n_polys * sizeof(bool));

    cudaMemcpy(d_dist, dist, n_polys * n_rays * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points, n_polys * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_verts, verts, n_rays * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces, n_faces * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, n_polys * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_polys + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = n_rays * n_polys * sizeof(float) + 3 * n_polys * sizeof(float) + 3 * n_rays * sizeof(float) + 3 * n_faces * sizeof(int) + n_polys * sizeof(float);
    cuda_non_max_suppression_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_dist, d_points, d_verts, d_faces, d_scores, d_result, n_polys, n_rays, n_faces, threshold, use_bbox, use_kdtree, verbose);

    cudaMemcpy(result, d_result, n_polys * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_dist);
    cudaFree(d_points);
    cudaFree(d_verts);
    cudaFree(d_faces);
    cudaFree(d_scores);
    cudaFree(d_result);
#else
    // CPU parallelism with OpenMP
    cpu_non_max_suppression(dist, points, verts, faces, scores, result, n_polys, n_rays, n_faces, threshold, use_bbox, use_kdtree, verbose);
#endif

    return PyArray_Return(arr_result);
}

static PyMethodDef methods[] = {
    {"c_non_max_suppression_inds", c_non_max_suppression_inds, METH_VARARGS, "Apply non-maximum suppression."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "nms_module",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_nms_module(void) {
    import_array();
    return PyModule_Create(&module);
}


//--------------------------------------------------------------

// dist.shape = (n_polys, n_rays)
// points.shape = (n_polys, 3)
// verts.shape = (n_rays, 3)
// faces.shape = (n_faces, 3)
// labels.shape = (n_polys,)
// shape = (3,)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept
//
// render_mode
// 0 -> full
// 1 -> kernel
// 2 -> bbox

static PyObject* c_polyhedron_to_label(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_points=NULL,*arr_verts=NULL,*arr_faces=NULL,*arr_labels=NULL;
  PyArrayObject *arr_result=NULL;

  int nz,ny,nx;
  int render_mode;
  int verbose;
  int use_overlap_label;
  int overlap_label;
  
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiii(iii)",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_points,
                        &PyArray_Type, &arr_verts,
                        &PyArray_Type, &arr_faces,
                        &PyArray_Type, &arr_labels,
                        &render_mode,
                        &verbose,
                        &use_overlap_label,
                        &overlap_label,
                        &nz,&ny,&nx))
    return NULL;


  const int n_polys = PyArray_DIMS(arr_dist)[0];
  const int n_rays = PyArray_DIMS(arr_dist)[1];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const points = (float*) PyArray_DATA(arr_points);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);
  const int * const labels = (int*) PyArray_DATA(arr_labels);


  npy_intp dims_result[3];
  dims_result[0] = nz;
  dims_result[1] = ny;
  dims_result[2] = nx;

  // result array 
  arr_result = (PyArrayObject*)PyArray_ZEROS(3,dims_result,NPY_INT32,0);


  int * result = (int*) PyArray_DATA(arr_result);
  
  _COMMON_polyhedron_to_label(dist, points,
                              verts,faces,
                              n_polys, n_rays, n_faces,
                              labels, 
                              nz, ny,nx,
                              render_mode,
                              verbose,
                              use_overlap_label,
                              overlap_label,  
                              result);
  
  
  return PyArray_Return(arr_result);
}

//------------------------------------------------------------------------


static PyObject* c_dist_to_volume(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_verts=NULL,*arr_faces=NULL;
  PyArrayObject *arr_result=NULL;


  if (!PyArg_ParseTuple(args, "O!O!O!",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_verts,
                        &PyArray_Type, &arr_faces))
    return NULL;


  const int nz = PyArray_DIMS(arr_dist)[0];
  const int ny = PyArray_DIMS(arr_dist)[1];
  const int nx = PyArray_DIMS(arr_dist)[2];
  const int n_rays = PyArray_DIMS(arr_dist)[3];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);

  const float origin[3] = {0.0, 0.0, 0.0};


  npy_intp dims_result[3];
  dims_result[0] = nz;
  dims_result[1] = ny;
  dims_result[2] = nx;
  
  arr_result = (PyArrayObject*)PyArray_ZEROS(3,dims_result,NPY_FLOAT32,0);

  float * result = (float*) PyArray_DATA(arr_result);

  // ................................

  _COMMON_dist_to_volume(dist, origin, verts, faces, n_rays, n_faces, nx, ny, nz, result);

  // ................................

  return PyArray_Return(arr_result);
}

static PyObject* c_dist_to_centroid(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_verts=NULL,*arr_faces=NULL;
  PyArrayObject *arr_result=NULL;

  int absolute;

  if (!PyArg_ParseTuple(args, "O!O!O!i",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_verts,
						&PyArray_Type, &arr_faces,
						&absolute))
    return NULL;


  const int nz = PyArray_DIMS(arr_dist)[0];
  const int ny = PyArray_DIMS(arr_dist)[1];
  const int nx = PyArray_DIMS(arr_dist)[2];
  const int n_rays = PyArray_DIMS(arr_dist)[3];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);


  const float origin[3] = {0.0, 0.0, 0.0};


  npy_intp dims_result[4];
  dims_result[0] = nz;
  dims_result[1] = ny;
  dims_result[2] = nx;
  dims_result[3] = 3;

  arr_result = (PyArrayObject*)PyArray_ZEROS(4,dims_result,NPY_FLOAT32,0);

  float * result = (float*) PyArray_DATA(arr_result);


  _COMMON_dist_to_centroid(dist, origin, verts, faces, n_rays, n_faces, nx, ny, nz, absolute, result);

  return PyArray_Return(arr_result);
}



// // -----------------------------------------------------------------------


// create the star convex distances from a labeled image
static PyObject* c_star_dist3d(PyObject *self, PyObject *args) {

  PyArrayObject *src = NULL;
  PyArrayObject *dst = NULL;

  PyArrayObject *pdx = NULL;
  PyArrayObject *pdy = NULL;
  PyArrayObject *pdz = NULL;


  int n_rays;
  int grid_x, grid_y, grid_z;


  if (!PyArg_ParseTuple(args, "O!O!O!O!iiii", &PyArray_Type, &src, &PyArray_Type, &pdz ,&PyArray_Type, &pdy,&PyArray_Type, &pdx, &n_rays,&grid_z,&grid_y,&grid_x))
    return NULL;

  npy_intp *dims = PyArray_DIMS(src);

  npy_intp dims_dst[4];
  // the resulting shape with tuple(slice(0, None, g) for g in grid)
  dims_dst[0] = (dims[0]-1)/grid_z+1;
  dims_dst[1] = (dims[1]-1)/grid_y+1;
  dims_dst[2] = (dims[2]-1)/grid_x+1;
  dims_dst[3] = n_rays;

  dst = (PyArrayObject*)PyArray_SimpleNew(4,dims_dst,NPY_FLOAT32);

  // # pragma omp parallel for schedule(dynamic)
  // strangely, using schedule(dynamic) leads to segfault on OSX when importing skimage first 
#ifdef __APPLE__    
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic) 
#endif
  for (int i=0; i<dims_dst[0]; i++) {
    for (int j=0; j<dims_dst[1]; j++) {
      for (int k=0; k<dims_dst[2]; k++) {
        const unsigned short value = *(unsigned short *)PyArray_GETPTR3(src,i*grid_z,j*grid_y,k*grid_x);
        // background pixel
        if (value == 0) {
          for (int n = 0; n < n_rays; n++) {
            *(float *)PyArray_GETPTR4(dst,i,j,k,n) = 0;
          }
          // foreground pixel
        } else {

          for (int n = 0; n < n_rays; n++) {

            float dx = *(float *)PyArray_GETPTR1(pdx,n);
            float dy = *(float *)PyArray_GETPTR1(pdy,n);
            float dz = *(float *)PyArray_GETPTR1(pdz,n);

            // dx /= 4;
            // dy /= 4;
            // dz /= 4;
            float x = 0, y = 0, z=0;
            // move along ray
            while (1) {
              x += dx;
              y += dy;
              z += dz;
              const int ii = round_to_int(i*grid_z+z), jj = round_to_int(j*grid_y+y), kk = round_to_int(k*grid_x+x);

              //std::cout<<"ii: "<<ii<<" vs  "<<i*grid_z+z<<std::endl;

              // stop if out of bounds or reaching a pixel with a different value/id
              if (ii < 0 || ii >= dims[0] ||
                  jj < 0 || jj >= dims[1] ||
                  kk < 0 || kk >= dims[2] ||
                  value != *(unsigned short *)PyArray_GETPTR3(src,ii,jj,kk))
                {
                  const int x2 = round_to_int(x);
                  const int y2 = round_to_int(y);
                  const int z2 = round_to_int(z);
                  const float dist = sqrt(x2*x2 + y2*y2 + z2*z2);
                  *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;

                  // const float dist = sqrt(x*x + y*y + z*z);
                  // *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;

                  // small correction as we overshoot the boundary
                  // const float t_corr = .5f/fmax(fmax(fabs(dx),fabs(dy)),fabs(dz));
                  // printf("%.2f\n", t_corr);
                  // x += (t_corr-1.f)*dx;
                  // y += (t_corr-1.f)*dy;
                  // z += (t_corr-1.f)*dz;
                  // const float dist = sqrt(x*x + y*y + z*z);
                  // *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;
                  
                  break;
                }
            }
          }
        }

      }
    }
  }

  return PyArray_Return(dst);
}


//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
                                       {"c_star_dist3d",
                                        c_star_dist3d,
                                        METH_VARARGS,
                                        "star dist 3d calculation"},
                                       
                                       {"c_non_max_suppression_inds",
                                        c_non_max_suppression_inds,
                                        METH_VARARGS,
                                        "non-maximum suppression"},
                                       
                                       {"c_polyhedron_to_label",
                                        c_polyhedron_to_label,
                                        METH_VARARGS,
                                        "polyhedron to label"},
                                       
                                       {"c_dist_to_volume",
                                        c_dist_to_volume,
                                        METH_VARARGS,
                                        "distance to volume"},
                                       
                                       {"c_dist_to_centroid",
                                        c_dist_to_centroid,
                                        METH_VARARGS,
                                        "distance to centroids"},

                                       {NULL, NULL, 0, NULL}                                       
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "stardist3d", 
                                       NULL,         
                                       -1,           
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_stardist3d(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
