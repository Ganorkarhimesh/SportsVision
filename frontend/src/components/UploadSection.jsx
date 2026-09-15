"use client";

import { useRef, useState } from "react";
import {
  Upload,
  Video,
  LoaderCircle,
  X,
  AlertCircle
} from "lucide-react";

import { predictVideo } from "@/lib/api";
import VideoResult from "./VideoResult";

export default function UploadSection() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const selectFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("video/")) {
      setError("Please upload a valid video file.");
      return;
    }

    setError("");
    setResult(null);
    setFile(selectedFile);

    const videoURL = URL.createObjectURL(selectedFile);
    setPreview(videoURL);
  };

  const handleFileChange = (event) => {
    selectFile(event.target.files[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);

    const droppedFile = event.dataTransfer.files[0];
    selectFile(droppedFile);
  };

  const removeFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const analyzeVideo = async () => {
    if (!file) return;

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const response = await predictVideo(file);

      setResult(response);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="upload-section" id="upload">
      <div className="section-heading">
        <p className="section-label">VIDEO ANALYSIS</p>

        <h2>
          Upload your <span>sports video</span>
        </h2>

        <p>
          SportsVision will detect the athlete, extract spatial features,
          analyze temporal movement and predict the action.
        </p>
      </div>

      {!file && (
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">
            <Upload size={30} />
          </div>

          <h3>Drop your sports video here</h3>

          <p>or click to browse from your computer</p>

          <span>MP4, AVI, MOV</span>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            hidden
          />
        </div>
      )}

      {file && (
        <div className="video-upload-card">
          <div className="video-preview">
            <video src={preview} controls />
          </div>

          <div className="file-info">
            <Video size={20} />

            <div>
              <strong>{file.name}</strong>
              <p>{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>

            <button onClick={removeFile} className="icon-button">
              <X size={20} />
            </button>
          </div>

          <button
            className="analyze-button"
            onClick={analyzeVideo}
            disabled={loading}
          >
            {loading ? (
              <>
                <LoaderCircle className="spin" size={20} />
                Processing Video...
              </>
            ) : (
              "Analyze Video"
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="error-message">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      {loading && (
        <div className="processing-message">
          <LoaderCircle className="spin" size={22} />

          <div>
            <strong>SportsVision is analyzing your video...</strong>
            <p>
              YOLOv8 → ResNet-50 → Bi-LSTM → Self-Attention
            </p>
          </div>
        </div>
      )}

      {result && <VideoResult result={result} />}
    </section>
  );
}