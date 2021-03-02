// -*- C++ -*-
//
// EnergyEnergyCorrelators - Evaluates EECs on particle physics events
// Copyright (C) 2020-2021 Patrick T. Komiske III
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

%{
// needed by numpy.i, harmless otherwise
#define SWIG_FILE_WITH_INIT

// standard library headers we need
#include <cstdlib>
#include <cstring>
%}

// include numpy typemaps
%include numpy.i
%init %{
import_array();
%}

%pythoncode %{
import numpy as _np
%}

%define %additional_numpy_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE)

/* Typemap suite for (DATA_TYPE** ARGOUTVIEWM_ARRAY6, DIM_TYPE* DIM1, DIM_TYPE* DIM2,
                      DIM_TYPE* DIM3, DIM_TYPE* DIM4, DIM_TYPE* DIM5, DIM_TYPE* DIM6)
 */
%typemap(in,numinputs=0)
  (DATA_TYPE** ARGOUTVIEWM_ARRAY6, DIM_TYPE* DIM1    , DIM_TYPE* DIM2    , DIM_TYPE* DIM3    , DIM_TYPE* DIM4    , DIM_TYPE* DIM5    , DIM_TYPE* DIM6    )
  (DATA_TYPE* data_temp = NULL   , DIM_TYPE dim1_temp, DIM_TYPE dim2_temp, DIM_TYPE dim3_temp, DIM_TYPE dim4_temp, DIM_TYPE dim5_temp, DIM_TYPE dim6_temp)
{
  $1 = &data_temp;
  $2 = &dim1_temp;
  $3 = &dim2_temp;
  $4 = &dim3_temp;
  $5 = &dim4_temp;
  $6 = &dim5_temp;
  $7 = &dim6_temp;
}
%typemap(argout,
         fragment="NumPy_Backward_Compatibility,NumPy_Utilities")
  (DATA_TYPE** ARGOUTVIEWM_ARRAY6, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4, DIM_TYPE* DIM5, DIM_TYPE* DIM6)
{
  npy_intp dims[6] = { *$2, *$3, *$4, *$5, *$6, *$7 };
  PyObject* obj = PyArray_SimpleNewFromData(6, dims, DATA_TYPECODE, (void*)(*$1));
  PyArrayObject* array = (PyArrayObject*) obj;

  if (!array) SWIG_fail;

%#ifdef SWIGPY_USE_CAPSULE
    PyObject* cap = PyCapsule_New((void*)(*$1), SWIGPY_CAPSULE_NAME, free_cap);
%#else
    PyObject* cap = PyCObject_FromVoidPtr((void*)(*$1), free);
%#endif

%#if NPY_API_VERSION < 0x00000007
  PyArray_BASE(array) = cap;
%#else
  PyArray_SetBaseObject(array,cap);
%#endif

  $result = SWIG_Python_AppendOutput($result,obj);
}

%enddef // additional_numpy_typemaps

%additional_numpy_typemaps(double, NPY_DOUBLE, int)

// numpy typemaps
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {
  (double** arr_out0, int* n0),
  (double** arr_out1, int* n1)
}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
  (double** arr_out0, int* n0, int* n1)
}
%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {
  (double** arr_out0, int* n0, int* n1, int* n2),
  (double** arr_out1, int* m0, int* m1, int* m2)
}
%apply (double** ARGOUTVIEWM_ARRAY6, int* DIM1, int* DIM2, int* DIM3, int* DIM4, int* DIM5, int* DIM6) {
  (double** arr_out0, int* n0, int* n1, int* n2, int* n3, int* n4, int* n5)
}

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_DOUBLE_ARRAY(arr_out, n, size, nbytes)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

// mallocs a 3D array of doubles of the specified size
%define MALLOC_2D_DOUBLE_ARRAY(arr_out, n0, n1, size0, size1, nbytes)
  *n0 = size0;
  *n1 = size1;
  size_t nbytes = size_t(*n0)*size_t(*n1)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

// mallocs a 3D array of doubles of the specified size
%define MALLOC_3D_DOUBLE_ARRAY(arr_out, n0, n1, n2, size0, size1, size2, nbytes)
  *n0 = size0;
  *n1 = size1;
  *n2 = size2;
  size_t nbytes = size_t(*n0)*size_t(*n1)*size_t(*n2)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

// mallocs a 6D array of doubles of the specified size
%define MALLOC_6D_DOUBLE_ARRAY(arr_out, n0, n1, n2, n3, n4, n5, size0, size1, size2, size3, size4, size5, nbytes)
  *n0 = size0;
  *n1 = size1;
  *n2 = size2;
  *n3 = size3;
  *n4 = size4;
  *n5 = size5;
  size_t nbytes = size_t(*n0)*size_t(*n1)*size_t(*n2)*size_t(*n3)*size_t(*n4)*size_t(*n5)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL)
    throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes");
%enddef

%define COPY_1DARRAY_TO_NUMPY(arr_out0, n0, size, nbytes, ptr)
  MALLOC_1D_DOUBLE_ARRAY(arr_out0, n0, size, nbytes)
  memcpy(*arr_out0, ptr, nbytes);
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR(pyname, cppname, size)
void pyname(double** arr_out0, int* n0) {
  COPY_1DARRAY_TO_NUMPY(arr_out0, n0, size, nbytes, $self->cppname().data())
}
%enddef
