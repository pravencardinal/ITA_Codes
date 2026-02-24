from airflow import DAG
from airflow.operators.email_operator import EmailOperator
from airflow.operators.python_operator import PythonOperator
from airflow import models
from datetime import timedelta, date, timezone, datetime
from airflow.utils.dates import days_ago, parse_execution_date
from typing import Dict, List, Optional, Sequence, Tuple, Union
from google.cloud import aiplatform, aiplatform_v1
from google.oauth2 import service_account
from airflow.models import Variable
import json
import time
import warnings
import os, argparse
import asyncio
from google.cloud import aiplatform
from typing import Optional
from pytz import timezone
import logging


# These args will get passed on to each operator
default_args = {
        'owner'                 : 'medai',
        'description'           : 'The is an example DAG for Vertex AI custom job in EDDIE 2.0',
        'depend_on_past'        : False,
        'start_date'            : datetime(2021, 9, 14),
        'email_on_failure'      : False,
        'email_on_retry'        : False,
        'retries'               : 1,
        'retry_delay'           : timedelta(minutes=5),
        'depends_on_past'       : False
}


dag_name = "eddie_med_ai_pwr_prediction"
custom_job_name= "eddie_med_ai_pwr_prediction_custom"

# Set environment
eddie_env=Variable.get("eddie_env")
# TRAINING update variables for training environment 
#eddie_env = "TRN" # hard coded for testing
if eddie_env == "TRN":
    schedule_interval = "0 8 * * 1"
    project = 'edd-trn-mdai-pr-cah'
    service_account='sa-eddie-med-ai-pwr@edd-trn-mdai-pr-cah.iam.gserviceaccount.com'
    container_uri='gcr.io/edd-trn-cde-pr-cah/med-ai-pwr:0.0.4'
    #args = ["-GCS_BUCKET", "cah-eddie-trn-med-ai-cxpa-np", "-EDNA_PROJECT", "edna-datastg-pr-cah","-EDDIE_PROJECT", "edd-trn-mdai-pr-cah","-MODEL_OUT_DIR", "MODEL_ARTIFACTS_DIR"]
    machine_type = 'n1-standard-4'
    bucket_name = 'cah-eddie-trn-med-ai-pwr-np'
    destination_blob_name = 'cah-eddie-trn-med-ai-pwr-np/harbor'
    BUCKET_URI = f"gs://{bucket_name}/{destination_blob_name}"
    staging_bucket = f"{BUCKET_URI}"
    location = 'us-central1'
    #email_list = ['sagar.virmani@cardinalhealth.com']
    email_to = "g-ds-global-usmpd-mktg-sourcing@cardinalhealth.com"
# STAGE update variables for stage environment 
elif eddie_env == "STG":
    schedule_interval = "0 8 * * 1"
    project = 'eddie-stg-pr-cah'
    service_account = 'sa-eddie-med-ai-pwr@eddie-stg-pr-cah.iam.gserviceaccount.com'
    container_uri = 'gcr.io/eddie-stg-pr-cah/med-ai-pwr:stg-0.0.5'
    # args = ["-GCS_BUCKET", "cah-eddie-trn-med-ai-cxpa-np", "-EDNA_PROJECT", "edna-datastg-pr-cah","-EDDIE_PROJECT", "edd-trn-mdai-pr-cah","-MODEL_OUT_DIR", "MODEL_ARTIFACTS_DIR"]
    machine_type = 'n1-standard-4'
    bucket_name = 'cah-eddie-stg-med-ai-pwr-np'
    destination_blob_name = 'cah-eddie-stg-med-ai-pwr-np/models'
    BUCKET_URI = f"gs://{bucket_name}/{destination_blob_name}"
    staging_bucket = f"{BUCKET_URI}"
    location = 'us-central1'
    # email_list = ['sagar.virmani@cardinalhealth.com']
    email_to = "g-ds-global-usmpd-mktg-sourcing@cardinalhealth.com"
# copy and update above values
# PRODUCTION  update variables for production environment 
elif eddie_env == "PRD":
    schedule_interval = "0 8 * * 1"
    project = 'eddie-pr-cah'
    service_account = 'sa-eddie-med-ai-pwr@eddie-pr-cah.iam.gserviceaccount.com'
    container_uri = 'gcr.io/eddie-pr-cah/med-ai-pwr:prd-0.0.7'
    # args = ["-GCS_BUCKET", "cah-eddie-trn-med-ai-cxpa-np", "-EDNA_PROJECT", "edna-datastg-pr-cah","-EDDIE_PROJECT", "edd-trn-mdai-pr-cah","-MODEL_OUT_DIR", "MODEL_ARTIFACTS_DIR"]
    machine_type = 'n1-standard-4'
    bucket_name = 'cah-eddie-prd-med-ai-pwr-np'
    destination_blob_name = 'cah-eddie-prd-med-ai-pwr-np/models'
    BUCKET_URI = f"gs://{bucket_name}/{destination_blob_name}"
    staging_bucket = f"{BUCKET_URI}"
    location = 'us-central1'
    # email_list = ['sagar.virmani@cardinalhealth.com']
    email_to = "g-ds-global-usmpd-mktg-sourcing@cardinalhealth.com"
# copy and update above values


worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": machine_type,
#                         "accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
#                         "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": container_uri,
            "command": ["python","main.py"],
            #"args": args
        },
    }
]

def vertex_ai_custom_job ():
    custom_job = aiplatform.CustomJob(
        display_name=custom_job_name,
        worker_pool_specs=worker_pool_specs,
        project = project,
        staging_bucket=staging_bucket,
        location = location,
        labels={'ml_use_case': 'pwr','apmid':'31441','costcenter':'2060000932'},
    )
    custom_job.run(
        service_account= service_account
    )
        

with models.DAG(dag_name,
                default_args = default_args,
                schedule_interval = schedule_interval,
                catchup = False,
                access_control={"medai": {"can_read", "can_edit"}},
                max_active_runs=1,
                dagrun_timeout=timedelta(minutes=180)
) as dag:


    vertex_ai_run_job = PythonOperator(
        task_id = 'vertex_ai_custom_job_task',
        python_callable = vertex_ai_custom_job,
        dag = dag)

    success_task_id = "success_mail"
    subject = "Vertex AI pipeline - " + custom_job_name
    html_content = "<p> Airflow triggered Vertex AI pipeline : " + custom_job_name +"  </p>"

    success = EmailOperator(
        task_id = success_task_id, 
        to = email_to,
        subject = subject,
        html_content = html_content,
        dag = dag)

    task_id = "error_email"
    trigger_rule = "one_failed"
    subject = "Error Triggering Vertex AI pipeline - " + custom_job_name
    html_content = "<p> Airflow failed to trigger Vertex AI pipeline : " + custom_job_name + " </p>"

    error_email = EmailOperator(
            task_id = task_id,
            trigger_rule = trigger_rule,
            to = email_to,
            subject = subject,
            html_content = html_content,
            dag = dag)

    vertex_ai_run_job >> success
    vertex_ai_run_job >> error_email
