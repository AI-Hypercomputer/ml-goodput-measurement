<!--
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

 # Overview

 Cloud TPU Goodput is a library intended to be used with Cloud TPU to log the
 necessary information and query a job's Goodput. It can be pip installed to
 import its modules, and retrieve information about a training job's overall
 productive Goodput. The package exposes API interfaces to log useful
 information from the user application and query Goodput for the job run, gain
 insight into the productivity of ML workloads and utilization of compute
 resources.