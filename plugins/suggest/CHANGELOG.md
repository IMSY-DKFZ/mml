# Changelog

All notable changes to this plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.1 (12/19/2024):
Public release version.

## 0.3.0 (08/29/2024):
This release has renamed this plugin from `mml-inference` to `mml-suggest` and drastically refactored functionality.

### Features
 - the procedure of bleuprint compilation has been completely refactored:
   - use `suggest` mode to generate blueprints 
   - reuse multiple distance measures and determine distance preference relations per pipeline key
   - cutoffs ensure top pipelines to be leveraged
   - temperature allows to introduce randomness
   - automatic inclusion of existing tasks - no need to compile task_list yourself
