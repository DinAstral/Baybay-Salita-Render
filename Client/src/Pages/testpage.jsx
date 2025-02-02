import React, { useState, useRef } from "react";

const SpeechToTextApp = () => {
  const [recording, setRecording] = useState(false);
  const [resultText, setResultText] = useState("");
  const [language, setLanguage] = useState("en-US");
  const [audioBlob, setAudioBlob] = useState(null);
  const recordBtnRef = useRef(null);
  const resultRef = useRef(null);
  const recognitionRef = useRef(null);
  const [interimText, setInterimText] = useState("");
  const [isDownloadDisabled, setIsDownloadDisabled] = useState(true);

  const languages = [
    { no: "18", name: "Filipino", native: "Filipino", code: "tl-PH" },
  ];

  const speechToText = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = language;
    recognition.interimResults = true;
    recognitionRef.current = recognition;

    recordBtnRef.current.classList.add("recording");
    recordBtnRef.current.querySelector("p").innerHTML = "Listening...";

    recognition.start();

    let audioChunks = [];
    recognition.onresult = (event) => {
      const speechResult = event.results[0][0].transcript;

      if (event.results[0].isFinal) {
        setResultText((prevResult) => prevResult + " " + speechResult);
        setInterimText("");
        // Convert audio to blob and set state
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        setAudioBlob(audioBlob);
        uploadAudio(audioBlob);
      } else {
        setInterimText(speechResult);
      }
      setIsDownloadDisabled(false);
    };

    recognition.onend = () => {
      stopRecording();
    };

    recognition.onerror = (event) => {
      stopRecording();

      if (event.error === "no-speech") {
        alert("No speech was detected. Stopping...");
      } else if (event.error === "audio-capture") {
        alert(
          "No microphone was found. Ensure that a microphone is installed."
        );
      } else if (event.error === "not-allowed") {
        alert("Permission to use microphone is blocked.");
      } else if (event.error === "aborted") {
        alert("Listening Stopped.");
      } else {
        alert("Error occurred in recognition: " + event.error);
      }
    };
  };

  const stopRecording = () => {
    recognitionRef.current.stop();
    recordBtnRef.current.querySelector("p").innerHTML = "Start Listening";
    recordBtnRef.current.classList.remove("recording");
    setRecording(false);
  };

  const handleRecordClick = () => {
    if (!recording) {
      speechToText();
      setRecording(true);
    } else {
      stopRecording();
    }
  };

  const uploadAudio = (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");

    fetch("/api/upload-audio", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Speech recognition result:", data);
        setResultText(data.userSpeechText);
        setIsDownloadDisabled(false);
      })
      .catch((error) => {
        console.error("Error uploading audio:", error);
      });
  };

  const downloadText = () => {
    const text = resultRef.current.innerText;
    const filename = "speech.txt";

    const element = document.createElement("a");
    element.setAttribute(
      "href",
      "data:text/plain;charset=utf-8," + encodeURIComponent(text)
    );
    element.setAttribute("download", filename);
    element.style.display = "none";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const clearText = () => {
    setResultText("");
    setInterimText("");
    setIsDownloadDisabled(true);
  };

  return (
    <div>
      <div>
        <select
          id="language"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
        >
          {languages.map((lang) => (
            <option key={lang.no} value={lang.code}>
              {lang.name} ({lang.native})
            </option>
          ))}
        </select>
      </div>

      <button className="record" ref={recordBtnRef} onClick={handleRecordClick}>
        <p>Start Listening</p>
      </button>

      <div className="result" ref={resultRef}>
        <p>{resultText}</p>
        {interimText && <p className="interim">{interimText}</p>}
      </div>

      <button
        className="download"
        onClick={downloadText}
        disabled={isDownloadDisabled}
      >
        Download
      </button>

      <button className="clear" onClick={clearText}>
        Clear
      </button>
    </div>
  );
};

export default SpeechToTextApp;
