import { useState, useEffect } from "react";
import { Button, Tooltip } from "@nextui-org/react";
import profile from "./../../assets/profile.png";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faAddressCard,
  faCircleInfo,
  faEnvelope,
} from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

const BodyAdminViewTeacher = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { UserID } = useParams();

  const [activeIndex, setActiveIndex] = useState(0);
  const [profileImgUrl, setProfileImgUrl] = useState(profile);

  const [data, setData] = useState({
    _id: "",
    UserID: "",
    FirstName: "",
    LastName: "",
    Section: "",
    Department: "",
    Age: "",
    Birthday: "",
    Gender: "",
    Address: "",
    Status: "",
    Nationality: "",
    ContactNumber: "",
    email: "",
  });

  useEffect(() => {
    axios
      .get(`/api/getTeacher/${UserID}`)
      .then((response) => {
        setData(response.data);
        setProfileImgUrl(response.data.Picture || profile);
      })
      .catch((err) => {
        toast.error("Failed to fetch teacher data. Please try again later.");
      });
  }, [UserID]);

  useEffect(() => {
    const menuItem = [`/adminViewTeacher/${data?.UserID}`];
    const index = menuItem.findIndex((item) => item === location.pathname);
    setActiveIndex(index);
  }, [data?.UserID, location.pathname]);

  const toggleActive = (index) => {
    setActiveIndex(index);
  };

  return (
    <div className="p-10">
      <div className="flex items-center justify-start gap-2 mb-5 ">
        <h1 className="text-3xl font-semibold">View Teacher Profile</h1>
        <Tooltip
          showArrow={true}
          content={
            <div className="p-2">
              <div className="text-sm font-bold">View Information</div>
              <div className="text-xs">
                This function will view the teacher's information in the system.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
        </Tooltip>
      </div>
      <div className="w-full max-w-8xl mx-auto bg-white shadow-lg rounded-lg p-[3rem]">
        {/* Profile Header */}
        <div className="flex flex-col md:flex-row items-center md:items-start gap-6">
          <img
            src={profileImgUrl}
            className="w-32 h-32 rounded-full"
            alt="Profile"
          />
          <div className="flex-1">
            <h2 className="text-2xl font-bold">
              {data.FirstName && data.LastName
                ? `${data.FirstName} ${data.LastName}`
                : "Teacher Name"}
            </h2>
            <h3>
              {data.Section ? `Section: ${data.Section}` : "Teacher of section"}
            </h3>
            <div className="flex flex-col mt-4 text-gray-700">
              <div className="flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faAddressCard}
                  className="text-gray-600"
                />
                <span>Teacher ID Number: {data.UserID || "N/A"}</span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <FontAwesomeIcon icon={faEnvelope} className="text-gray-600" />
                <span>Email Address: {data.email || "N/A"}</span>
              </div>
            </div>
          </div>
          <Button
            color="primary"
            className="w-[120px] text-md p-1"
            onClick={() => navigate("/AdminUsers")}
          >
            Back
          </Button>
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
              Basic Information
            </Button>
            <Button
              variant="light"
              radius="none"
              className={`text-md font-medium ${
                activeIndex === 1 ? "border-b-4 border-blue-500" : ""
              }`}
              onClick={() => toggleActive(1)}
            >
              Education Attainment
            </Button>
          </div>

          {/* Content Based on Active Tab */}
          {activeIndex === 0 && (
            <div className="mt-6 bg-[#faf9f4] p-6 rounded-lg shadow">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <span className="block text-sm text-gray-600">
                    First Name:
                  </span>
                  <p className="text-gray-800">{data.FirstName || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Last Name:
                  </span>
                  <p className="text-gray-800">{data.LastName || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Civil Status:
                  </span>
                  <p className="text-gray-800">{data.Status || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Gender:</span>
                  <p className="text-gray-800">{data.Gender || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Birthday:</span>
                  <p className="text-gray-800">{data.Birthday || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Contact Number:
                  </span>
                  <p className="text-gray-800">{data.ContactNumber || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Address:</span>
                  <p className="text-gray-800">{data.Address || "N/A"}</p>
                </div>
              </div>
            </div>
          )}

          {activeIndex === 1 && (
            <div className="mt-6 bg-[#f9f9f9] p-6 rounded-lg shadow">
              <p>Education Attainment information goes here...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BodyAdminViewTeacher;
