use pyo3::prelude::*;
use polars::prelude::*;
use rand::Rng;
use rand::RngExt;
use std::time::Instant;
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use ndarray::{ArrayD, Axis, IxDyn};
use ndarray::Array2;
use pyo3::exceptions::PyValueError;
use numpy::PyArray2;
use numpy::PyReadonlyArray2;

#[pyfunction]
fn filter_df(x: i32) -> PyResult<f64> {
    let mut rng = rand::rng();

    let col1: Vec<i32> = (0..x).map(|_| rng.random_range(0..=1)).collect();
    let col2: Vec<i32> = (0..x).map(|_| rng.random_range(0..=1)).collect();
    let col3: Vec<i32> = (0..x).map(|_| rng.random_range(0..=1)).collect();
    let col4: Vec<i32> = (0..x).map(|_| rng.random_range(0..=1)).collect();
    let col5: Vec<i32> = (0..x).map(|_| rng.random_range(0..=1)).collect();

    let df = df![
        "col1" => col1,
        "col2" => col2,
        "col3" => col3,
        "col4" => col4,
        "col5" => col5
    ]
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let start = Instant::now();

    let col1 = df
        .column("col1")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .i32()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mask = col1.equal(1);

    let filtered_df = df
        .filter(&mask)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    //Ok(())
    Ok(start.elapsed().as_secs_f64())
}

#[pyfunction]
fn filter_df_given_df<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, i64>,
    filter_col: usize,
) -> PyResult<Bound<'py, PyArrayDyn<i64>>> {
    let arr = input.as_array();
    let ncols = arr.ncols();

    if filter_col >= ncols {
        return Err(PyValueError::new_err("filter_col out of bounds"));
    }

    let data = input
        .as_slice()
        .map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;

    let mut output = Vec::<i64>::with_capacity(data.len() / 2);

    for row in data.chunks_exact(ncols) {
        if row[filter_col] == 1 {
            output.extend_from_slice(row);
        }
    }

    let output_rows = output.len() / ncols;

    let result = Array2::from_shape_vec(
        (output_rows, ncols),
        output,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Array2<i64> -> ArrayD<i64> -> PyArrayDyn<i64>
    Ok(result.into_dyn().into_pyarray(py))
}

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pymodule]
fn dask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(filter_df, m)?)?;
    m.add_function(wrap_pyfunction!(filter_df_given_df, m)?)?;

    Ok(())
}
