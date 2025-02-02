import React, { useState, useEffect, useContext } from "react";
import { Button, Tooltip } from "@nextui-org/react";
import profile from "./../../assets/profile.png";
import { useLocation, useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAddressCard, faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import { UserContext } from "../../../context/userContext";
import axios from "axios";
import toast from "react-hot-toast";

const LazyProfileImage = ({ src, alt }) => (
  <img loading="lazy" src={src} alt={alt} className="w-32 h-32 rounded-full" />
);

const BodyInformationKid = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
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
  const [profileImgUrl, setProfileImgUrl] = useState(profile);

  useEffect(() => {
    axios
      .get(`/api/getParent/${user.UserID}`)
      .then((response) => {
        setParent(response.data);
      })
      .catch((err) => setError("Failed to load parent data."))
      .finally(() => setLoading(false));
  }, [user.UserID]);

  useEffect(() => {
    const LRN = parent?.Student?.[0]?.LRN;

    if (LRN) {
      setLoading(true);
      axios
        .get(`/api/getStudent/${LRN}`)
        .then((response) => {
          setStudent(response.data);
          setProfileImgUrl(response.data.Picture || profile);
        })
        .catch((err) => setError("Failed to load student data."))
        .finally(() => setLoading(false));
    }
  }, [parent]);

  const handleTabChange = (index) => setActiveIndex(index);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append("Profile", file);
      formData.append("role", student.role);
      formData.append("LRN", student.LRN);

      try {
        const response = await axios.post("/api/profileUpdate", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        if (response.data.message) {
          setProfileImgUrl(response.data.updatedProfileUrl);
          toast.success("Profile picture updated successfully!");
        } else if (response.data.error) {
          toast.error(`Error: ${response.data.error}`);
        }
      } catch (error) {
        toast.error(
          "Failed to update profile picture. Please try again later."
        );
      }
    }
  };

  return (
    <div className="p-10">
      <div className="flex items-center justify-start gap-2 mb-5">
        <h1 className="text-3xl font-semibold">Student Information</h1>
        <Tooltip
          showArrow
          content={
            <div className="p-2">
              <div className="text-sm font-bold">View Information</div>
              <div className="text-xs">
                This function allows you to view the student's information in
                the system.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
        </Tooltip>
      </div>
      {error && <p className="text-red-600 mt-4">{error}</p>}
      {loading ? (
        <div className="flex justify-center">
          <p>Loading...</p>
        </div>
      ) : (
        <div className="w-full max-w-8xl mx-auto bg-white shadow-lg rounded-lg p-[3rem]">
          {/* Profile Header */}
          <div className="flex flex-col md:flex-row items-center md:items-start gap-6">
            <Tooltip
              content={
                <div className="p-2">
                  <div className="text-sm font-bold">
                    Student Picture Update
                  </div>
                  <div className="text-xs">
                    You can update profile picture once clicked.
                  </div>
                </div>
              }
            >
              <label className="cursor-pointer">
                <img
                  src={profileImgUrl}
                  className="w-32 h-32 rounded-full"
                  alt="Profile"
                />
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                  aria-label="Update profile picture"
                />
              </label>
            </Tooltip>
            <div className="flex-1">
              <h2 className="text-2xl font-bold">{`${student.FirstName} ${student.LastName}`}</h2>
              <h3>{`Section: ${student.Section}`}</h3>
              <div className="flex flex-col mt-4 text-gray-700">
                <div className="flex items-center gap-2">
                  <FontAwesomeIcon
                    icon={faAddressCard}
                    className="text-gray-600"
                  />
                  <span>Learner Reference Number: {student.LRN || "N/A"}</span>
                </div>
                {/* Student Status Highlight */}
                <div className="mt-4">
                  <span className="text-sm font-bold">Status:</span>
                  <span
                    className={`ml-2 px-2 py-1 rounded-full ${
                      student.Status === "Grade Ready Reader"
                        ? "bg-green-200 text-green-800"
                        : student.Status === "Transitioning Reader"
                        ? "bg-blue-200 text-blue-800"
                        : student.Status === "Developing Reader"
                        ? "bg-yellow-200 text-yellow-800"
                        : student.Status === "Incomplete"
                        ? "bg-gray-200 text-gray-800"
                        : "bg-red-200 text-red-800"
                    }`}
                  >
                    {student.Status || "N/A"}
                  </span>
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
                onClick={() => handleTabChange(0)}
              >
                Basic Information
              </Button>
            </div>
            {activeIndex === 0 && (
              <div className="mt-6 bg-[#faf9f4] p-6 rounded-lg shadow">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <span className="block text-sm text-gray-600">
                      First Name:
                    </span>
                    <p className="text-gray-800">
                      {student.FirstName || "N/A"}
                    </p>
                  </div>
                  <div>
                    <span className="block text-sm text-gray-600">
                      Last Name:
                    </span>
                    <p className="text-gray-800">{student.LastName || "N/A"}</p>
                  </div>
                  <div>
                    <span className="block text-sm text-gray-600">Gender:</span>
                    <p className="text-gray-800">{student.Gender || "N/A"}</p>
                  </div>
                  <div>
                    <span className="block text-sm text-gray-600">
                      Birthday:
                    </span>
                    <p className="text-gray-800">{student.Birthday || "N/A"}</p>
                  </div>
                  <div>
                    <span className="block text-sm text-gray-600">
                      Address:
                    </span>
                    <p className="text-gray-800">{student.Address || "N/A"}</p>
                  </div>
                  <div>
                    <span className="block text-sm text-gray-600">
                      Contact Number:
                    </span>
                    <p className="text-gray-800">
                      {student.ContactNumber || "N/A"}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default BodyInformationKid;
