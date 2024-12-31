# Changelog

All notable changes to this plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.2 (12/19/2024):
Public release version.

## 0.5.1 (11/15/2024):
Minor patch to reflect the removal of the `plotting` config group in `mml-core`

## 0.5.0 (08/29/2024):
This release comprises a major refactoring of the similarity plugin to allow more increasing knowledge style, better 
reusability and a focus on the pivot task.

### Features
 - automatic reuse of existing `features` and `fim`
 - non necessarily predict all (n x n) distances but allow (1 x n) calculations by setting a pivot task
 - easier inheriting of abstract task distance scheduler by implementing a default distance routine
