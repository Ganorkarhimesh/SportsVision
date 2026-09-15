"use client";

import {
  CheckCircle2,
  UserRound,
  Clock,
  Film,
  Brain
} from "lucide-react";

export default function VideoResult({ result }) {
  const confidence = Number(result.confidence || 0);

  return (
    <div className="result-container">
      <div className="result-header">
        <div>
          <p className="section-label">ANALYSIS COMPLETE</p>
          <h2>Prediction Result</h2>
        </div>

        <CheckCircle2 size={30} />
      </div>

      {result.processed_video_url && (
        <div className="processed-video">
          <video
            src={result.processed_video_url}
            controls
            playsInline
          />
        </div>
      )}

      <div className="prediction-card">
        <p>Predicted Action</p>

        <h3>{result.action || "Unknown Action"}</h3>

        <div className="confidence-header">
          <span>Confidence</span>
          <strong>{confidence.toFixed(2)}%</strong>
        </div>

        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${Math.min(confidence, 100)}%` }}
          />
        </div>
      </div>

      <div className="result-grid">
        <div className="result-stat">
          <UserRound size={22} />

          <div>
            <span>Athlete Detected</span>
            <strong>
              {result.detected_athlete ? "Yes" : "No"}
            </strong>
          </div>
        </div>

        <div className="result-stat">
          <Brain size={22} />

          <div>
            <span>YOLOv8 Confidence</span>
            <strong>
              {Number(result.yolo_confidence || 0).toFixed(2)}%
            </strong>
          </div>
        </div>

        <div className="result-stat">
          <Film size={22} />

          <div>
            <span>Frames Processed</span>
            <strong>{result.frames || 0}</strong>
          </div>
        </div>

        <div className="result-stat">
          <Clock size={22} />

          <div>
            <span>Processing Time</span>
            <strong>
              {Number(result.processing_time || 0).toFixed(2)} sec
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
}