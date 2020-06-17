from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI

# Utility Functions

@pytest.fixture(params=[pytest.param("interval", marks=pytest.mark.xfail(reason="swarm not implemented in 1d")),
                        "square",
                        pytest.param("extruded", marks=pytest.mark.xfail(reason="extruded meshes not supported")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        pytest.param("periodicrectangle", marks=pytest.mark.xfail(reason="meshes made from coordinate fields are not supported"))])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(1, 1), 1)
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


def pseudo_random_coords(size):
    """
    Get an array of pseudo random coordinates with coordinate elements
    between -0.5 and 1.5. The random numbers are consistent for any
    given `size` since `numpy.random.seed(0)` is called each time this
    is used.
    """
    np.random.seed(0)
    a, b = -0.5, 1.5
    return (b - a) * np.random.random_sample(size=size) + a


# Function Space Generation Tests

def functionspace_tests(vm, family, degree):
    # Prep: Get number of cells
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(vm.num_cells(), op=MPI.SUM)
    # Can create function space
    V = FunctionSpace(vm, family, degree)
    # Can create function on function spaces
    f = Function(V)
    g = Function(V)
    # Can interpolate and Galerkin project onto functions
    gdim = vm.geometric_dimension()
    if gdim == 1:
        x, = SpatialCoordinate(vm)
        f.interpolate(x)
        g.project(x)
    elif gdim == 2:
        x, y = SpatialCoordinate(vm)
        f.interpolate(x*y)
        g.project(x*y)
    elif gdim == 3:
        x, y, z = SpatialCoordinate(vm)
        f.interpolate(x*y*z)
        g.project(x*y*z)
    # Get exact values at coordinates with maintained ordering
    assert np.shape(f.dat.data_ro)[0] == np.shape(vm.coordinates.dat.data_ro)[0]
    assert np.allclose(f.dat.data_ro, np.prod(vm.coordinates.dat.data_ro, 1))
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times
    f.interpolate(Constant(2))
    assert np.isclose(assemble(f*dx), 2*num_cells_mpi_global)


def vectorfunctionspace_tests(vm, family, degree):
    # Prep: Get number of cells
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(vm.num_cells(), op=MPI.SUM)
    # Can create function space
    V = VectorFunctionSpace(vm, family, degree)
    # Can create functions on function spaces
    f = Function(V)
    g = Function(V)
    # Can interpolate and Galerkin project onto functions
    x = SpatialCoordinate(vm)
    f.interpolate(2*as_vector(x))
    g.project(2*as_vector(x))
    # Get exact values at coordinates with maintained ordering
    assert np.shape(f.dat.data_ro)[0] == np.shape(vm.coordinates.dat.data_ro)[0]
    assert np.allclose(f.dat.data_ro, 2*vm.coordinates.dat.data_ro)
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times. Note that we get a vertex cell for
    # each geometric dimension so we have to sum over geometric
    # dimension too.
    gdim = vm.geometric_dimension()
    if gdim == 1:
        f.interpolate(Constant((1,)))
    if gdim == 2:
        f.interpolate(Constant((1, 1)))
    if gdim == 3:
        f.interpolate(Constant((1, 1, 1)))
    assert np.isclose(assemble(inner(f, f)*dx), num_cells_mpi_global*gdim)


@pytest.fixture(params=["DG", pytest.param("CG", marks=pytest.mark.xfail(reason="unsupported family"))])
def family(request):
    return request.param


@pytest.fixture(params=[0, pytest.param(1, marks=pytest.mark.xfail(reason="unsupported degree"))])
def degree(request):
    return request.param


def test_functionspaces(parentmesh, vertexcoords, family, degree):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    functionspace_tests(vm, family, degree)
    vectorfunctionspace_tests(vm, family, degree)


@pytest.mark.parallel
def test_functionspaces_parallel(parentmesh, vertexcoords, family, degree):
    test_functionspaces(parentmesh, vertexcoords, family, degree)
