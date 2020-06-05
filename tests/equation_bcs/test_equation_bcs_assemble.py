import pytest

from firedrake import *


def test_equation_bcs_direct_assemble_one_form():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = - inner(grad(u), grad(v)) * dx
    F1 = inner(u, v) * ds(1)
    bc = EquationBC(F1 == 0, u, 1)

    assemble(F, bcs=extract_equation_bc_forms(bc, 'F'))
    assemble(F, bcs=bc)


def test_equation_bcs_direct_assemble_two_form():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = - inner(grad(u), grad(v)) * dx
    a1 = inner(u, v) * ds(1)
    L1 = inner(Constant(0), v) * ds(1)
    u = Function(V)
    bc = EquationBC(a1 == L1, u, 1)

    assemble(a, bcs=extract_equation_bc_forms(bc, 'J'))
    assemble(a, bcs=extract_equation_bc_forms([bc], 'Jp'))
    with pytest.raises(RuntimeError) as excinfo:
        assemble(a, bcs=bc)
    assert "Unable to infer 'form_type' to be used ('J' or 'Jp')" in str(excinfo.value)
