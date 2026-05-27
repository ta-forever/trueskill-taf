// pytafskill — Python binding for the TAF team-TrueSkill + σ-relaxation rating engine.
//
// Exposes one function:
//   rate(env, pid1, pid2, score12, tstart, game_class, ratings, isHuman)
//     env         : (nGameClasses, ENV_SIZE)        float64
//     pid1, pid2  : (nGames, MAX_TEAMSIZE)          int32   — player ids per team; pad
//                                                              unused slots with -1.
//                                                              Both arrays must have the
//                                                              same -1 pattern per row,
//                                                              and that defines per-row
//                                                              teamSize.
//     score12     : (nGames,)                       int32   — +1 if team0 won, −1 if team1 won
//     tstart      : (nGames,)                       float64 — game start times (units of Δt)
//     game_class  : (nGames,)                       int32   — index into env's first axis
//     ratings     : (nPlayers, RATING_SIZE)         float64 — IN/OUT, mutated in place
//     isHuman     : (nPlayers,)                     bool    — true for human players
//   Returns
//     L          : (nGames,)                              float64 — P(outcome | ratings before this game)
//     nGames1    : (nGames, MAX_TEAMSIZE)                 int32   — games played by team1 player before this one
//     nGames2    : (nGames, MAX_TEAMSIZE)                 int32
//     r1         : (nGames, MAX_TEAMSIZE, RATING_SIZE)    float64 — pre-game ratings of team1 players
//     r2         : (nGames, MAX_TEAMSIZE, RATING_SIZE)    float64 — pre-game ratings of team2 players

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tafskill/TafskillFactorGraph.h"

namespace {

using tafskill::TafskillFactorGraph;

template <typename T>
bool check_dtype(PyArrayObject *arr, int npy_type, const char *name)
{
    if (PyArray_TYPE(arr) != npy_type)
    {
        PyErr_Format(PyExc_TypeError, "argument '%s' has wrong dtype (expected numpy type code %d, got %d)",
                     name, npy_type, PyArray_TYPE(arr));
        return false;
    }
    if (PyArray_STRIDE(arr, PyArray_NDIM(arr) - 1) != sizeof(T))
    {
        PyErr_Format(PyExc_ValueError, "argument '%s' must be C-contiguous along the last axis", name);
        return false;
    }
    return true;
}

PyObject *rate(PyObject * /*self*/, PyObject *args)
{
    PyArrayObject *env_obj      = nullptr;
    PyArrayObject *pid1_obj     = nullptr;
    PyArrayObject *pid2_obj     = nullptr;
    PyArrayObject *score12_obj  = nullptr;
    PyArrayObject *tstart_obj   = nullptr;
    PyArrayObject *gclass_obj   = nullptr;
    PyArrayObject *ratings_obj  = nullptr;
    PyArrayObject *isHuman_obj  = nullptr;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!",
                          &PyArray_Type, &env_obj,
                          &PyArray_Type, &pid1_obj,
                          &PyArray_Type, &pid2_obj,
                          &PyArray_Type, &score12_obj,
                          &PyArray_Type, &tstart_obj,
                          &PyArray_Type, &gclass_obj,
                          &PyArray_Type, &ratings_obj,
                          &PyArray_Type, &isHuman_obj))
    {
        return nullptr;
    }

    if (!check_dtype<double>(env_obj,     NPY_DOUBLE, "env"))      return nullptr;
    if (!check_dtype<int>(   pid1_obj,    NPY_INT32,  "pid1"))     return nullptr;
    if (!check_dtype<int>(   pid2_obj,    NPY_INT32,  "pid2"))     return nullptr;
    if (!check_dtype<int>(   score12_obj, NPY_INT32,  "score12"))  return nullptr;
    if (!check_dtype<double>(tstart_obj,  NPY_DOUBLE, "tstart"))   return nullptr;
    if (!check_dtype<int>(   gclass_obj,  NPY_INT32,  "game_class")) return nullptr;
    if (!check_dtype<double>(ratings_obj, NPY_DOUBLE, "ratings"))  return nullptr;
    if (PyArray_TYPE(isHuman_obj) != NPY_BOOL) {
        PyErr_SetString(PyExc_TypeError, "isHuman must be bool"); return nullptr;
    }

    if (PyArray_NDIM(pid1_obj) != 2 || PyArray_NDIM(pid2_obj) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "pid1 and pid2 must be 2-D (nGames, teamSize)");
        return nullptr;
    }
    const npy_intp nGames    = PyArray_DIM(pid1_obj, 0);
    const npy_intp maxTeamSz = PyArray_DIM(pid1_obj, 1);
    if (PyArray_DIM(pid2_obj, 0) != nGames || PyArray_DIM(pid2_obj, 1) != maxTeamSz)
    {
        PyErr_SetString(PyExc_ValueError, "pid1 and pid2 must have matching shape");
        return nullptr;
    }
    if (PyArray_NDIM(ratings_obj) != 2 || PyArray_DIM(ratings_obj, 1) != TafskillFactorGraph::RatingSize())
    {
        PyErr_Format(PyExc_ValueError, "ratings must have shape (nPlayers, %d)",
                     TafskillFactorGraph::RatingSize());
        return nullptr;
    }
    if (PyArray_NDIM(env_obj) != 2 || PyArray_DIM(env_obj, 1) != TafskillFactorGraph::EnvSize())
    {
        PyErr_Format(PyExc_ValueError, "env must have shape (nGameClasses, %d)",
                     TafskillFactorGraph::EnvSize());
        return nullptr;
    }

    const npy_intp nPlayers = PyArray_DIM(ratings_obj, 0);

    // Output arrays
    npy_intp Ldims[]  = { nGames };
    npy_intp NGdims[] = { nGames, maxTeamSz };
    npy_intp Rdims[]  = { nGames, maxTeamSz, TafskillFactorGraph::RatingSize() };
    PyArrayObject *L_obj       = (PyArrayObject *)PyArray_ZEROS(1, Ldims,  NPY_DOUBLE, 0);
    PyArrayObject *nGames1_obj = (PyArrayObject *)PyArray_ZEROS(2, NGdims, NPY_INT32,  0);
    PyArrayObject *nGames2_obj = (PyArrayObject *)PyArray_ZEROS(2, NGdims, NPY_INT32,  0);
    PyArrayObject *r1_obj      = (PyArrayObject *)PyArray_ZEROS(3, Rdims,  NPY_DOUBLE, 0);
    PyArrayObject *r2_obj      = (PyArrayObject *)PyArray_ZEROS(3, Rdims,  NPY_DOUBLE, 0);
    if (!L_obj || !nGames1_obj || !nGames2_obj || !r1_obj || !r2_obj)
    {
        Py_XDECREF(L_obj); Py_XDECREF(nGames1_obj); Py_XDECREF(nGames2_obj);
        Py_XDECREF(r1_obj); Py_XDECREF(r2_obj);
        return nullptr;
    }

    // Per-player bookkeeping
    std::vector<int>    gameCount(nPlayers, 0);
    std::vector<double> lastGameTime(nPlayers, -1.0);

    // Cache one factor graph per (game-class, teamSize)
    std::unordered_map<long long, std::shared_ptr<TafskillFactorGraph>> graphs;

    try
    {
        for (npy_intp n = 0; n < nGames; ++n)
        {
            const int *pids1   = (int *)PyArray_GETPTR1(pid1_obj,   n);
            const int *pids2   = (int *)PyArray_GETPTR1(pid2_obj,   n);
            const int  score12 = *(int *)PyArray_GETPTR1(score12_obj, n);
            const int  gclass  = *(int *)PyArray_GETPTR1(gclass_obj,  n);
            const double tstart= *(double *)PyArray_GETPTR1(tstart_obj, n);

            if (gclass < 0 || gclass >= PyArray_DIM(env_obj, 0))
            {
                std::ostringstream s;
                s << "game " << n << " has game_class=" << gclass << " out of range";
                throw std::runtime_error(s.str());
            }
            const double *env = (double *)PyArray_GETPTR2(env_obj, gclass, 0);

            // Derive this game's effective teamSize from non-(-1) entries in pids1.
            int teamSize = 0;
            for (npy_intp j = 0; j < maxTeamSz; ++j)
            {
                if (pids1[j] >= 0) ++teamSize;
                else break;
            }
            if (teamSize <= 0)
            {
                std::ostringstream s;
                s << "game " << n << " has zero-size team";
                throw std::runtime_error(s.str());
            }

            const long long key = ((long long)gclass << 32) | (long long)teamSize;
            auto it = graphs.find(key);
            if (it == graphs.end())
            {
                graphs[key] = std::make_shared<TafskillFactorGraph>(env, teamSize, 0);
                it = graphs.find(key);
            }
            TafskillFactorGraph &graph = *it->second;

            // PreGameSetup with current ratings
            for (int iPlayer = 0; iPlayer < teamSize; ++iPlayer)
            {
                const int p1 = pids1[iPlayer];
                const int p2 = pids2[iPlayer];
                if (p1 < 0 || p1 >= nPlayers || p2 < 0 || p2 >= nPlayers)
                {
                    std::ostringstream s;
                    s << "game " << n << " has out-of-range pid (" << p1 << "," << p2 << ")";
                    throw std::runtime_error(s.str());
                }

                const bool isHuman1 = *(npy_bool *)PyArray_GETPTR1(isHuman_obj, p1) != 0;
                const bool isHuman2 = *(npy_bool *)PyArray_GETPTR1(isHuman_obj, p2) != 0;
                (void)isHuman1; (void)isHuman2; // unused (no separate AI model in tafskill)

                const double dt1 = (lastGameTime[p1] > 0.0) ? (tstart - lastGameTime[p1]) : 0.0;
                const double dt2 = (lastGameTime[p2] > 0.0) ? (tstart - lastGameTime[p2]) : 0.0;
                lastGameTime[p1] = tstart;
                lastGameTime[p2] = tstart;

                *(int *)PyArray_GETPTR2(nGames1_obj, n, iPlayer) = gameCount[p1]++;
                *(int *)PyArray_GETPTR2(nGames2_obj, n, iPlayer) = gameCount[p2]++;

                const double *r1 = (double *)PyArray_GETPTR2(ratings_obj, p1, 0);
                const double *r2 = (double *)PyArray_GETPTR2(ratings_obj, p2, 0);

                // Record pre-game ratings (before this game updates them) for plotting.
                for (int k = 0; k < TafskillFactorGraph::RatingSize(); ++k)
                {
                    *(double *)PyArray_GETPTR3(r1_obj, n, iPlayer, k) = r1[k];
                    *(double *)PyArray_GETPTR3(r2_obj, n, iPlayer, k) = r2[k];
                }

                graph.PreGameSetup(0, iPlayer, r1, dt1, true);
                graph.PreGameSetup(1, iPlayer, r2, dt2, true);
            }

            double pwin = 0.0, pdraw = 0.0, plose = 0.0;
            graph.ComputePriors(pwin, pdraw, plose);

            const double Ln = (score12 > 0) ? pwin : (score12 < 0 ? plose : pdraw);
            *(double *)PyArray_GETPTR1(L_obj, n) = Ln;

            graph.ComputePosteriors(score12);

            for (int iPlayer = 0; iPlayer < teamSize; ++iPlayer)
            {
                const int p1 = pids1[iPlayer];
                const int p2 = pids2[iPlayer];
                if (p1 != p2)
                {
                    graph.ReadoutPosteriors(0, iPlayer, (double *)PyArray_GETPTR2(ratings_obj, p1, 0));
                    graph.ReadoutPosteriors(1, iPlayer, (double *)PyArray_GETPTR2(ratings_obj, p2, 0));
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(L_obj); Py_DECREF(nGames1_obj); Py_DECREF(nGames2_obj);
        Py_DECREF(r1_obj); Py_DECREF(r2_obj);
        return nullptr;
    }

    return Py_BuildValue("(NNNNN)",
                         (PyObject *)L_obj,
                         (PyObject *)nGames1_obj, (PyObject *)nGames2_obj,
                         (PyObject *)r1_obj,      (PyObject *)r2_obj);
}

PyObject *env_size(PyObject *, PyObject *)    { return PyLong_FromLong(TafskillFactorGraph::EnvSize()); }
PyObject *rating_size(PyObject *, PyObject *) { return PyLong_FromLong(TafskillFactorGraph::RatingSize()); }

PyMethodDef Methods[] = {
    {"rate",        rate,        METH_VARARGS, "Run team-TrueSkill + σ-relaxation over a series of games"},
    {"env_size",    env_size,    METH_NOARGS,  "Number of environment parameters per game class"},
    {"rating_size", rating_size, METH_NOARGS,  "Number of per-player rating parameters (μ, σ²)"},
    {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT, "pytafskill", "", -1, Methods,
    nullptr, nullptr, nullptr, nullptr};

} // anonymous namespace

PyMODINIT_FUNC PyInit_pytafskill(void)
{
    import_array();
    if (PyErr_Occurred()) return nullptr;
    return PyModule_Create(&ModuleDef);
}
