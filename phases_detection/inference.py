import cv2
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
import torch
from utils import is_pixelated, calculate_red_percentage, accumulator_post_processing, convert_seconds_to_timestamp
import numpy as np
from MSTCN2 import Evaluator
import argparse
import pandas as pd


def extract_embeddings(video_path, model_name, transforms, device, sample_rate=5, target_fps=15):
    # Load the trained ResNet model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # Adjust output layer for the number of classes
    # load trained weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_name))
    else:
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
    feature_extractor.to(device)
    video_features = []
    ts = 0
    timestamps = []
    previous_class = ''
    in_body = []
    frame_numbers = []
    with torch.no_grad():
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int((original_fps*sample_rate) // target_fps)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                model_out_of_body = YOLO('deidentification.pt')
                results = model_out_of_body(frame, verbose=False, conf=0.9)[0]
                probs = results.probs  # Classification probabilities for each detected class
                predicted_class = model_out_of_body.names[
                    int(probs.top1)]  # Get the class with the highest probability

                if predicted_class == 'Ex Situ' and previous_class == 'Ex Situ' and calculate_red_percentage(frame) < 20:
                    in_body.append(False)
                elif is_pixelated(frame):
                    in_body.append(False)
                else:
                    in_body.append(True)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply the transform to the frame
                transformed_frame = transforms(frame_rgb)
                transformed_frame = transformed_frame.to(device)
                batch_frame = transformed_frame.unsqueeze(0)
                features = feature_extractor(batch_frame).squeeze()  # [batch_size, 2048, 1, 1] -> [batch_size, 2048]
                features = features.cpu().tolist()
                video_features.append(features)
                    
                timestamps.append(ts)
                frame_numbers.append(frame_count+1)
                previous_class = predicted_class
            frame_count+=1
            ts+=1
            
        cap.release()
        cv2.destroyAllWindows()
    timestamps = [ts/original_fps for ts in timestamps]
    return np.array(video_features), timestamps, in_body, frame_numbers


def inference(embeddings, model_name, in_body):
    # initialize mstcn model params
    phases_dict = {
        "nasal": 0,
        "sphenoid": 1,
        "sellar": 2,
        "closure": 3
    }
    
    ordered_list_procedure_breakdown = [k for k in phases_dict.keys()]
    num_classes = len(ordered_list_procedure_breakdown)
    features_dim = 2048
    num_layers_PG = 11
    num_layers_R = 10
    num_R = 3
    num_f_maps = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_rate = 1

    filtered_predictions = embeddings[in_body]

    mstcn_evaluator = Evaluator(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes)

    mstcn_predictions = mstcn_evaluator.predict(model_name, filtered_predictions, phases_dict, device, sample_rate)
    
    # replace out of body predictions with the last predicted phase.
    last_prediction = mstcn_predictions[0]
    predictions = []
    pred_idx = 0

    for body in in_body:
        if not body:
            predictions.append(last_prediction)
        else:
            predictions.append(mstcn_predictions[pred_idx])
            last_prediction = mstcn_predictions[pred_idx]
            pred_idx += 1

    thresh = len(predictions) // 100
    
    mstcn_thresholds_dict_proportional = {cls: int((thresh + (thresh*(predictions.count(cls) / len(predictions))))) for cls in ordered_list_procedure_breakdown}

    diverge_thresh = 25*len(predictions) // 100

    lowest_thresh = 5*len(predictions) // 100

    accumulator_predicted = accumulator_post_processing(predictions, mstcn_thresholds_dict_proportional, ordered_list_procedure_breakdown, diverge_thresh, lowest_thresh)
    return accumulator_predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs inference on a video")
    parser.add_argument('--video_path', help='full path to pts video', type=str)
    parser.add_argument('--resnet_model_checkpoint', help='full path to resnet model weights', type=str, default='resnet50_feature_extractor_3_8_2024.pth')
    parser.add_argument('--mstcn_model_checkpoint', help='full path to mstcn model weights', type=str, default='mstcn_10_8_2024_2.pth')

    args, _ = parser.parse_known_args()

    rn_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.20258744060993195, 0.09945355355739594, 0.09271615743637085], 
                                 std=[0.23958955705165863, 0.15955227613449097, 0.15576229989528656]),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings, timestamps, in_body, frame_numbers = extract_embeddings(args.video_path, args.resnet_model_checkpoint, rn_transforms, device)
    predictions = inference(embeddings, args.mstcn_model_checkpoint, in_body)

    df = pd.DataFrame()
    df['Phase'] = [pred.capitalize() for pred in predictions]
    df["Timestamp"] = [convert_seconds_to_timestamp(ts) for ts in timestamps]
    df["Frame number"] = frame_numbers
    df.to_csv('predictions.csv')




