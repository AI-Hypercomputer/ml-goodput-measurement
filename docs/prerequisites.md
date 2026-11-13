# Prerequisites

The library stores its entries in Google Cloud Logging. This page covers
the project setup, the access scopes the nodes need, and the package
install.

## Google Cloud project

The usage of this package requires the setup of a Google Cloud project with
billing enabled to properly use Google Cloud Logging. If you don't have a Google
Cloud project, or if you don't have billing enabled for your Google Cloud
project, then do the following:

1. In the Google Cloud console, on the project selector page,
 [select or create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

2. Make sure that billing is enabled for your Google Cloud project. Instructions can be found [here](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console)

3. [Enable](https://console.cloud.google.com/flows/enableapi?apiid=logging.googleapis.com&_ga=2.27841276.1571868865.1726250448-123998259.1726107009) the Cloud Logging API.

  To run your training on Cloud accelerator, set up the environment by following
  instructions [here](https://cloud.google.com/tpu/docs/setup-gcp-account).

  To learn more about Google Cloud Logging, visit this [page](https://cloud.google.com/logging/docs).

## Access Scopes

 You will need both read and write access scopes for cloud logging on both the
 GPU or TPU and CPU node pools. Full cloud logging access is granted by the
 following access scope during node pool creation:

  - `https://www.googleapis.com/auth/cloud-platform`

   XPK adds this access scope to the GPU, TPU and CPU node pools, so XPK is the
   recommended method to create clusters and node-pools in you intend to run
   your workloads on GKE.

   Instructions on how to create clusters using XPK can be
   found [here](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md#cluster-create)
   and how to create workloads using XPK can be found
   [here](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md#workload-create).

   > **_NOTE:_** Access Scopes are immutable and workloads can only be migrated
  to new node pools with required access scopes. Access scopes on already
  created clusters cannot be updated.

## Installation

 To install the ML Goodput Measurement package, run the following command on the
 VM or machine you want to query or monitor your workload from:

 ```bash
 pip install ml-goodput-measurement
 ```

See also: [Recording intervals](recording.md), [Monitoring](monitoring.md).
