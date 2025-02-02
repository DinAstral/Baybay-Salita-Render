import { useState, useEffect, useContext } from "react";
import { Button, Tooltip } from "@nextui-org/react";
import { useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCircleInfo,
  faCheckCircle,
  faTimesCircle,
} from "@fortawesome/free-solid-svg-icons";

import { UserContext } from "../../../context/userContext";
import axios from "axios";
import toast from "react-hot-toast";
import AddFeedback from "../Modals/AddFeedback";

// Utility function to convert seconds to MM:SS format
const formatTimeRead = (secondsString) => {
  const seconds = parseInt(secondsString, 10);
  if (isNaN(seconds)) return "Not applicable";

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return `${minutes}:${
    remainingSeconds < 10 ? "0" : ""
  }${remainingSeconds} min`;
};

const BodyAdminViewPerformance = () => {
  const { UserInputId } = useParams();
  const navigate = useNavigate();
  const { user } = useContext(UserContext);

  const [modalShowView, setModalShowView] = useState(false);
  const [data, setData] = useState({
    UserInputId: "",
    LRN: "",
    Section: "",
    ActivityCode: "",
    Type: "",
    PerformanceItems: [],
    QuestionData: [],
    Score: "",
    TimeRead: "",
  });

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const response = await axios.get(`/api/getPerformance/${UserInputId}`);
        setData(response.data);
      } catch (err) {
        toast.error(
          "Failed to fetch performance data. Please try again later."
        );
      }
    };

    fetchPerformanceData();
  }, [UserInputId]);

  const handleViewClick = () => {
    setModalShowView(true);
  };

  return (
    <div className="px-9">
      <AddFeedback
        show={modalShowView}
        onHide={() => setModalShowView(false)}
        actCode={data.ActivityCode}
        userid={user?.UserID || ""}
        lrn={data.LRN}
        section={data.Section}
      />
      <div className="flex items-center justify-start gap-2 mb-5">
        <h1 className="text-3xl font-semibold">View Performance</h1>
        <Tooltip
          showArrow
          content={
            <div className="p-2">
              <div className="text-sm font-bold">View Student Performance</div>
              <div className="text-xs">
                This section displays the student's performance.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
        </Tooltip>
      </div>

      <div className="w-full max-w-8xl mx-auto bg-white shadow-lg rounded-lg p-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">Assessment Details</h2>
          <p className="text-gray-700">
            <strong>Activity Code:</strong> {data.ActivityCode || "N/A"}
          </p>
          <p className="text-gray-700">
            <strong>LRN:</strong> {data.LRN || "N/A"}
          </p>
          <p className="text-gray-700">
            <strong>Section:</strong> {data.Section || "N/A"}
          </p>
          <p className="text-gray-700">
            <strong>Type:</strong> {data.Type || "N/A"}
          </p>
        </div>

        {/* Display Score for all assessments */}
        <div className="mb-6">
          <p className="text-gray-700">
            <strong>Score:</strong> {data.Score || "N/A"}
          </p>
        </div>

        {/* If the assessment type is 'Pagbabasa' */}
        {data.Type === "Pagbabasa" && (
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4">
              Questions and Answers
            </h2>
            <p className="text-gray-700 mb-2">
              <strong>Time Read:</strong> {formatTimeRead(data.TimeRead)}
            </p>

            {data.QuestionData && data.QuestionData.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {data.QuestionData.map((questionObj, index) => (
                  <div
                    key={index}
                    className={`p-4 bg-gray-50 rounded-lg shadow ${
                      questionObj.Remarks === "Correct"
                        ? "border-2 border-green-500"
                        : "border-2 border-red-500"
                    } relative`}
                  >
                    <p className="text-gray-800 flex justify-start gap-2">
                      <strong>Question {index + 1}:</strong>{" "}
                      {questionObj.Question || "N/A"}
                    </p>
                    <p className="text-gray-800 flex justify-start gap-7">
                      <strong>Answer:</strong> {questionObj.Answer || "N/A"}
                    </p>
                    <p className="text-gray-800 flex justify-start gap-5">
                      <strong>Remarks:</strong> {questionObj.Remarks || "N/A"}
                    </p>

                    {/* Check or Cross Icon */}
                    <FontAwesomeIcon
                      icon={
                        questionObj.Remarks === "Correct"
                          ? faCheckCircle
                          : faTimesCircle
                      }
                      className={`absolute top-2 right-2 ${
                        questionObj.Remarks === "Correct"
                          ? "text-green-500"
                          : "text-red-500"
                      }`}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-700">
                No questions available for this assessment.
              </p>
            )}
          </div>
        )}

        {/* Items section for non-Pagbabasa assessments */}
        {data.Type !== "Pagbabasa" && (
          <div>
            <h2 className="text-xl font-semibold">Items</h2>
            {data.PerformanceItems && data.PerformanceItems.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {data.PerformanceItems.map((item, index) => (
                  <div
                    key={index}
                    className={`p-4 bg-gray-50 rounded-lg shadow ${
                      item.Remarks === "Correct"
                        ? "border-2 border-green-500"
                        : "border-2 border-red-500"
                    } relative`}
                  >
                    <p className="text-gray-800">
                      <strong>Item Code:</strong> {item.ItemCode}
                    </p>
                    <p className="text-gray-800">
                      <strong>Word:</strong> {item.Word || "N/A"}
                    </p>
                    <p className="text-gray-800">
                      <strong>Remarks:</strong> {item.Remarks || "N/A"}
                    </p>

                    {/* Flex container for audio buttons */}
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        {item.UserAudioURL ? (
                          <Tooltip content="Play User Audio">
                            <Button
                              as="a"
                              href={item.UserAudioURL}
                              target="_blank"
                              rel="noopener noreferrer"
                              color="primary"
                              size="sm"
                            >
                              Play User Audio
                            </Button>
                          </Tooltip>
                        ) : (
                          "No audio uploaded"
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        {item.DefaultAudio ? (
                          <Tooltip content="Play Default Audio">
                            <Button
                              as="a"
                              href={item.DefaultAudio}
                              target="_blank"
                              rel="noopener noreferrer"
                              color="primary"
                              size="sm"
                            >
                              Play Default Audio
                            </Button>
                          </Tooltip>
                        ) : (
                          "No default audio available"
                        )}
                      </div>
                      {/* Check or Cross Icon */}
                      <FontAwesomeIcon
                        icon={
                          item.Remarks === "Correct"
                            ? faCheckCircle
                            : faTimesCircle
                        }
                        className={`absolute top-2 right-2 ${
                          item.Remarks === "Correct"
                            ? "text-green-500"
                            : "text-red-500"
                        }`}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-700">
                No items available for this assessment.
              </p>
            )}
          </div>
        )}

        <div className="mt-6 flex gap-5">
          <Button color="danger" onClick={() => navigate(-1)} className="mt-4">
            Back
          </Button>
        </div>
      </div>
    </div>
  );
};

export default BodyAdminViewPerformance;
