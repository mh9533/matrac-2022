Projekt pri predmetu Matematika z računalnikom.

# Kalibracija meritev pametnih svetilk s strojnim učenjem

Supervised and unsupervised calibration of smart light sensor measurements

[Povezava](https://github.com/mare5x/ds-smart-lights) do glavnega repozitorija s kodo.

## Bližnjice

* Končno poročilo najdete [tukaj](./koncno_porocilo.pdf).
* Vmesno poročilo najdete [tukaj](./vmesno_porocilo.pdf).
* Python Jupyter notebooks so [tukaj](./src/notebooks/).  
* Izvorna koda je v mapi `./src/`

## Opis problema

Part of the [FRI Data Science Project Competition](https://datascience.fri.uni-lj.si/competition/). 
In collaboration with [Garex](https://www.garex.si/). 

> **Supervised and unsupervised calibration of smart light sensor measurements** (Garex)
> 
> One of core Garex’s projects is a system for smart lighting of sustainable cities of the future. Given access to historical data for a group of geographically close sensors and the ground truth, can we calibrate future measurements to improve their accuracy? For example, given temperature measurements for several smart lights on the same street and gold standard temperature measurements at that location, can we produce a model that is able to correct future sensor measurements so that they will be closer to the ground truth when the ground truth is not available? And, if ground truth is not available, can we at least detect outliers or improve measurements by smoothing over all sensors? In this project we will emphasize simplicity over complexity and robustness over incremental improvements in accuracy. The end goal is to produce a compact solution in Python that performs only these 2 tasks: (1) learning a calibration model for a group of sensors and (2) calibrating new observations given a learned model.
> 
> **Keywords**: sensor calibration, supervised and unsupervised learning, estimating measurement error
