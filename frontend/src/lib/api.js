const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function predictVideo(file) {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    let message = "Failed to process video.";

    try {
      const errorData = await response.json();
      message = errorData.detail || message;
    } catch {
      // Ignore JSON parsing error
    }

    throw new Error(message);
  }

  return response.json();
}