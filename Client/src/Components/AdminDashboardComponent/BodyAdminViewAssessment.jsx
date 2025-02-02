import { useState, useEffect } from "react";
import { Button, Tooltip } from "@nextui-org/react";
import { useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

const BodyAdminViewAssessment = () => {
  const navigate = useNavigate();
  const { ActivityCode } = useParams();

  // State for fetching assessment data
  const [data, setData] = useState({
    ActivityCode: "",
    Period: "",
    Type: "",
    Title: "",
    Sentence: "",
    Questions: [],
    Items: [],
  });

  // Fetch data on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`/api/getActivity/${ActivityCode}`);
        setData(response.data);
      } catch (err) {
        toast.error("Failed to fetch assessment data. Please try again later.");
      }
    };

    fetchData();
  }, [ActivityCode]);

  // Function to handle external redirects
  const handleRedirect = (url, isImage = false) => {
    if (isImage) {
      const img = new Image();
      img.src = url;
      const newTab = window.open("");
      newTab.document.write(img.outerHTML);
    } else {
      window.open(url, "_blank");
    }
  };

  return (
    <div className="px-9">
      <div className="flex items-center justify-start gap-2 mb-5">
        <h1 className="text-3xl font-semibold">View Assessment</h1>
        <Tooltip
          showArrow
          content={
            <div className="p-2">
              <div className="text-sm font-bold">View Information</div>
              <div className="text-xs">
                This section displays the assessment's information.
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
            <strong>Period:</strong> {data.Period || "N/A"}
          </p>
          <p className="text-gray-700">
            <strong>Type:</strong> {data.Type || "N/A"}
          </p>
        </div>

        {data.Type === "Pagbabasa" && (
          <div className="mb-6 bg-gray-100 p-4 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Reading Activity</h2>
            <p className="text-gray-800">
              <strong>Title:</strong> {data.Title || "N/A"}
            </p>
            <p className="text-gray-800">
              <strong>Sentence:</strong> {data.Sentence || "N/A"}
            </p>
          </div>
        )}

        {data.Type === "Pagbabasa" && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Questions</h2>
            {data.Questions.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {data.Questions.map((questionObj, qIndex) => (
                  <div
                    key={qIndex}
                    className="bg-gray-100 p-4 rounded-lg shadow"
                  >
                    <p className="text-gray-800">
                      <strong>Question {qIndex + 1}:</strong>{" "}
                      {questionObj.Question || "N/A"}
                    </p>
                    <p className="text-gray-800">
                      <strong>Answer:</strong> {questionObj.Answer || "N/A"}
                    </p>
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

        {data.Type !== "Pagbabasa" && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Items</h2>
            {data.Items.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {data.Items.map((item, index) => (
                  <div
                    key={index}
                    className="bg-gray-100 p-4 rounded-lg shadow"
                  >
                    <p className="text-gray-800">
                      <strong>Item Code:</strong> {item.ItemCode}
                    </p>
                    <p className="text-gray-800">
                      <strong>Word:</strong> {item.Word || "N/A"}
                    </p>

                    <div className="flex items-center space-x-4">
                      {item.Image && (
                        <Tooltip content="View Image">
                          <Button
                            color="primary"
                            size="sm"
                            className="my-2"
                            onPress={() =>
                              handleRedirect(item.SecureImage, true)
                            }
                          >
                            View Image
                          </Button>
                        </Tooltip>
                      )}
                      {item.Audio && (
                        <Tooltip content="Play Audio">
                          <Button
                            color="primary"
                            size="sm"
                            className="my-2"
                            onPress={() => handleRedirect(item.SecureAudio)}
                          >
                            Play Audio
                          </Button>
                        </Tooltip>
                      )}
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

        <div className="mt-6">
          <Button
            color="primary"
            onPress={() => navigate("/adminStudentAssessment")}
            className="my-4"
          >
            Back
          </Button>
        </div>
      </div>
    </div>
  );
};

export default BodyAdminViewAssessment;
