# Objective

Let’s assume you are setting up an OCR-based business process workflow to capture, annotate, identify, and classify images captured as part of an insurance product. To make sure that your model is performing well, there are several steps in this workflow that you’d like to observe. In this example, we pick one such step and create a continuous workflow that evaluates the OCR process.

# Goal

Typically, you would ingest images in a batch, transform them into encoded formats to consume for annotation. Let’s say you are reading in scanned documents and are looking for any text contained in the image. The first step is to ensure that the images themselves are of sufficient quality that you can trust the annotations that come out of this step. In this example, we observe the quality of images used during the OCR process.

# Monitoring image quality

Here, we will read in a batch of images that are presumed to be generated periodically. For each such set of images, we will compute a [Brisque score] (https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) and return the median of the scores to represent the quality of the dataset.When we detect a decrease in the score, we alert recipients of this issue.

# Continuous and proactive alerting

The following steps describe how to use Domino’s custom metrics framework to consume such batches of images, compute their quality index, and alert recipients on a periodic basis.

## 1. Define the metric
First, we will define the logic to compute this image quality index(median Brisque score). `image_quality.py` has the necessary definition of this process. It includes:

1.  Reading images from a public s3 bucket
    
2.  Computing individual Brisque scores for each image
    
3.  Computing a representative index for the set

## 2. Instrument with Custom Metrics Framework
Next, we will instrument the code to log the metrics computed to Domino so it can be retrieved at a later point for further analysis.
    

### A. Record each metric
Using the `log_metric` method included in the Domino SDK, we can record each metric and associated metadata against a registered model for monitoring. Domino will persist this for retrieval at any time.
    
### B. Evaluate against a threshold
In our example, we use a rolling window to compute the threshold to evaluate the index score against. To read back past results, we use the `read_metrics`  method in the SDK that lets us query a time-series results of logged metric values.

### C. Alert users
Now, we can check if the image quality index has deteriorated and alert the target recipients. Using the `trigger_alert` method in the SDK, we can inform users that a batch of images is of a quality standard that isn’t acceptable.

> **Note:** The list of recipients are retrieved from the Notifications section of the registered model in Domino Model Monitor.

They can now take remediative action to improve the OCR process for this batch before any incorrect or invalid decisions are made.    

### D. Deploy as a Scheduled Job
Finally, we deploy this into Domino’s Scheduled Jobs system for continuous evaluation.

1.  From your project, create a new Scheduled Job and provide the parameterized script as an input. In this case, you would use `image_quality.py` 
    
2.  Set up the required frequency to match the generation of a new dataset to compute the quality metrics for
    
3.  Deploy the scheduled job
    
4.  You can administer the monitoring job by visiting the Scheduled Jobs page in the project. This lets you observe basic operational metrics for each instance of the job execution. If any instance resulted in a threshold violation, you will see an email sent to the target recipients (this list is sourced from the registered model in Domino Model Monitor).

## 3. Analyze metrics
At any point, you can load the persisted metrics in a notebook inside a Domino workspace for a more detailed analysis. In our example, we will read back the logged image quality scores over time (using the `read_metrics` method) and compare two rolling window strategies to determine if there are outliers or if there needs to be an adjustment to the threshold. As you'll see in image_quality_sdk.ipynb, we plot the results for a 30-day and 7-day rolling window and find that the current threshold strategy is appropriate. 

**Note :**
In order to run the code in the example, the following libraries are needed and can be installed by adding the following lines to your environment's Dockerfile instructions

```
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install kats --user
RUN pip install plotly
RUN pip install piq --user
RUN pip install rfc3339
RUN pip install boto3
RUN pip install opencv-python
RUN pip install seaborn
RUN pip install iso8601
```
