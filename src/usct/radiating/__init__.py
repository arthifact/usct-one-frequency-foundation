"""Sim 2 -- radiating/impedance (velocity in -> pressure out): the INSTRUMENT-FACING forward.

This is the physically faithful model and the one the real instrument path is
built on: the transducer ring sits in an open water bath, outgoing waves radiate
away (no cavity resonances), and the observable is boundary pressure -- exactly
what a receiving transducer reads. The verified DtN sim (:mod:`usct.dtn`) remains
the reference the numerics are checked against.

Depends only on :mod:`usct.physics`; never imports :mod:`usct.dtn`.
"""
