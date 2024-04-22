import socket
import subprocess
import json
from django.http import JsonResponse
from django.shortcuts import render
from .forms import BusinessForm
from .forms import BusinessPredictionForm

def index(request):
    if request.method == 'POST':
        form = BusinessPredictionForm(request.POST)
        if form.is_valid():
            input_data = {
                "text": form.cleaned_data['text'],
                "total_hours_week": float(form.cleaned_data['total_hours_week']),
                "is_weekend_open": int(form.cleaned_data['is_weekend_open']),
                "state": form.cleaned_data['state'],
                "categories": form.cleaned_data['categories']
            }
            data_json = json.dumps(input_data)

            try:
                result = subprocess.run(
                    ['spark-submit', '/home/kokai1/scalable_folder/spark_model.py', data_json],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Filtering output to find JSON
                lines = result.stdout.split('\n')
                json_output = next((line for line in lines if line.strip().startswith('{')), None)

                if json_output:
                    prediction_output = json.loads(json_output)
                    # Get the last probability
                    data = json.loads(json_output)
                    probability_list = json.loads(data['probability'])
                    last_probability = round(probability_list[-1]*100, 2)
                else:
                    prediction_output = {'error': 'No valid JSON output received from Spark job', 'stderr': result.stderr}
                    last_probability = None
            except subprocess.CalledProcessError as e:
                prediction_output = {'error': str(e), 'stderr': e.stderr}
                last_probability = None
            except json.JSONDecodeError as e:
                prediction_output = {'error': 'JSON decoding error', 'details': str(e)}
                last_probability = None
        else:
            prediction_output = {}
            last_probability = None
    else:
        form = BusinessPredictionForm()
        prediction_output = {}
        last_probability = None

    return render(request, 'predictor/index.html', {'form': form, 'prediction': prediction_output, 'last_probability': last_probability})

def send_to_spark_and_receive(data):
    message = json.dumps(data)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 8765))
            s.sendall(message.encode())
            # Wait to receive data back from Spark
            response = s.recv(1024)  # Adjust buffer size as needed
    except Exception as e:
        return f"Failed to connect or receive data from Spark: {str(e)}"
    
    # Assuming the response is JSON and decode it
    try:
        prediction = json.loads(response.decode())
    except json.JSONDecodeError:
        prediction = "Invalid response format received from Spark"
    
    return prediction

def business_form(request):
    if request.method == 'POST':
        form = BusinessForm(request.POST)
        if form.is_valid():
            # Serialize form data to JSON
            data_json = json.dumps(form.cleaned_data)

            # Call the Spark script
            try:
                # Assuming 'spark_predict.py' is your Spark script ready to be executed
                # and it expects JSON data as input and returns JSON output
                result = subprocess.run(
                    ['spark-submit', '/home/kokai1/scalable_folder/spark_model.py', data_json],
                    capture_output=True, text=True, check=True
                )
                prediction_output = json.loads(result.stdout)

                # Add the prediction output to the context
                context = {'form': form, 'data': form.cleaned_data, 'prediction': prediction_output}
                return render(request, 'predictor/results.html', context)

            except subprocess.CalledProcessError as e:
                # Handle errors in Spark script execution
                return JsonResponse({'error': str(e), 'stderr': e.stderr}, status=500)

    else:
        form = BusinessForm()

    return render(request, 'predictor/form.html', {'form': form})


def predict_business(request):
    if request.method == 'POST':
        form = BusinessPredictionForm(request.POST)
        if form.is_valid():
            input_data = {
                "text": form.cleaned_data['text'],
                "total_hours_week": float(form.cleaned_data['total_hours_week']),
                "is_weekend_open": int(form.cleaned_data['is_weekend_open']),
                "state": form.cleaned_data['state'],
                "categories": form.cleaned_data['categories']
            }
            data_json = json.dumps(input_data)

            result = subprocess.run(
                ['spark-submit', '/home/kokai1/scalable_folder/spark_model.py', data_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            lines = result.stdout.split('\n')
            json_output = next((line for line in lines if line.strip().startswith('{')), None)

            if json_output:
                prediction_output = json.loads(json_output)
                # Get the last probability
                data = json.loads(json_output)
                probability_list = json.loads(data['probability'])
                last_probability = round(probability_list[-1]*100, 2)
                return render(request, 'predictor/results.html', {
                    'form': form,
                    'prediction': prediction_output,
                    'last_probability': last_probability  # Add this line
                })


    else:
        form = BusinessPredictionForm()

    return render(request, 'predictor/form.html', {'form': form})
