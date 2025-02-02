import React, { useState, useEffect, useContext } from "react";
import { Button, Tooltip } from "@nextui-org/react";
import profile from "./../../assets/profile.png";
import { useLocation, useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAddressCard, faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import { UserContext } from "../../../context/userContext";
import axios from "axios";
import toast from "react-hot-toast";

const BodyStudentProgress = () => {
  const [error, setError] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const { user } = useContext(UserContext);
  const [parent, setParent] = useState(null);
  const [student, setStudent] = useState({
    FirstName: "",
    LastName: "",
    Section: "",
    LRN: "",
    Nationality: "",
    Status: "",
    Gender: "",
    Birthday: "",
    Address: "",
    ContactNumber: "",
  });
  const [performance, setPerformance] = useState([]); // Added state for performance

  useEffect(() => {
    // Fetch parent data based on user.UserID
    const fetchParentData = async () => {
      try {
        const response = await axios.get(`/api/getParent/${user.UserID}`);
        setParent(response.data);
      } catch (err) {
        setError("Failed to load parent data.");
      }
    };

    if (user.UserID) {
      fetchParentData();
    }
  }, [user.UserID]);

  useEffect(() => {
    // Fetch student data based on parent data
    const fetchStudentData = async (LRN) => {
      try {
        const studentResponse = await axios.get(`/api/getStudentParent/${LRN}`);
        setStudent(studentResponse.data);

        const performanceResponse = await axios.get(
          `/api/getPerformanceStudent/${LRN}`
        );
        setPerformance(performanceResponse.data);
      } catch (err) {
        setError("Failed to load student data.");
      }
    };

    if (parent && parent.Student?.[0]?.LRN) {
      fetchStudentData(parent.Student[0].LRN);
    }
  }, [parent]);

  const toggleActive = (index) => {
    setActiveIndex(index);
  };

  return (
    <div className="p-10">
      <div className="flex items-center justify-start gap-2 mb-5">
        <h1 className="text-3xl font-semibold">Student Progress</h1>
        <Tooltip
          showArrow={true}
          content={
            <div className="p-2">
              <div className="text-sm font-bold">View Progress</div>
              <div className="text-xs">
                This function allows you to view the student's progress in the
                system.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
        </Tooltip>
      </div>

      <div className="w-full max-w-8xl mx-auto bg-white shadow-lg rounded-lg p-8">
        {/* Profile Header */}
        <div className="flex flex-col md:flex-row items-center md:items-start gap-6">
          <img src={profile} className="w-32 h-32 rounded-full" alt="Profile" />
          <div className="flex-1">
            <h2 className="text-2xl font-bold">
              {`${student.FirstName} ${student.LastName}`}
            </h2>
            <h4>{`Section: ${student.Section}`}</h4>
            <div className="flex flex-col mt-4 text-gray-700">
              <div className="flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faAddressCard}
                  className="text-gray-600"
                />
                <span>Learner Reference Number: {student.LRN}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="mt-8">
          <div className="flex gap-6 border-b pb-2">
            <Button
              variant="light"
              radius="none"
              className={`text-md font-medium ${
                activeIndex === 0 ? "border-b-4 border-blue-500" : ""
              }`}
              onClick={() => toggleActive(0)}
            >
              Progress
            </Button>
          </div>

          {/* Content Based on Active Tab */}
          {activeIndex === 0 && (
            <div className="mt-6 bg-[#f9f9f9] p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4">Student Progress</h2>
              {performance.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {performance.map((assessment, index) => (
                    <div
                      key={index}
                      className="p-4 bg-white rounded-lg shadow-md"
                    >
                      <div className="mb-2">
                        <span className="block text-sm text-gray-600">
                          Assessment: {assessment.Type || "N/A"}
                        </span>
                        <span className="block text-sm text-gray-600">
                          Activity Code: {assessment.ActivityCode || "N/A"}
                        </span>
                      </div>
                      <p className="text-gray-800">
                        Score: {assessment.Score || "N/A"}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-800">No performance data available.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BodyStudentProgress;
