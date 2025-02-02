import { useState, useEffect, useContext } from "react";
import profileImage from "./../../assets/profile.png";
import { useLocation, useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import {
  faAddressCard,
  faCircleInfo,
  faEnvelope,
} from "@fortawesome/free-solid-svg-icons";
import { UserContext } from "../../../context/userContext";

import axios from "axios";
import toast from "react-hot-toast";

const BodyTeacherProfileBasic = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeIndex, setActiveIndex] = useState(0);
  const { user } = useContext(UserContext);
  const [profileImg, setProfileImg] = useState(profileImage);
  const [data, setData] = useState({
    UserID: "",
    FirstName: "",
    LastName: "",
    Age: "",
    Section: "",
    Birthday: "",
    Gender: "",
    Address: "",
    Status: "",
    Nationality: "",
    ContactNumber: "",
    Picture: "",
    email: "",
  });

  useEffect(() => {
    axios
      .get(`/api/getTeacher/${user.UserID}`)
      .then((response) => {
        setData(response.data);
        setProfileImg(response.data.Picture || profileImage);
      })
      .catch(() => {
        toast.error("Failed to fetch teacher data. Please try again later.");
      });
  }, [user.UserID]);

  useEffect(() => {
    const menuItem = ["/teacherProfile"];
    const index = menuItem.findIndex((item) => item === location.pathname);
    setActiveIndex(index);
  }, [location.pathname]);

  // Calculate age based on birthday
  const calculateAge = (birthday) => {
    const birthDate = new Date(birthday);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDifference = today.getMonth() - birthDate.getMonth();
    if (
      monthDifference < 0 ||
      (monthDifference === 0 && today.getDate() < birthDate.getDate())
    ) {
      age--;
    }
    return age;
  };

  useEffect(() => {
    if (data.Birthday) {
      const age = calculateAge(data.Birthday);
      setData((prevData) => ({ ...prevData, Age: age }));
    }
  }, [data.Birthday]);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append("Profile", file);
      formData.append("role", user.role);
      formData.append("UserID", user.UserID);

      try {
        const response = await axios.post("/api/profileUpdate", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        if (response.data.message) {
          setProfileImg(response.data.updatedProfileUrl);
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

  const handleEditClick = () => {
    setModalShow(true);
  };

  const toggleActive = (index) => {
    setActiveIndex(index);
  };

  const handleUpdate = async (event) => {
    event.preventDefault(); // Prevent default form submission
    try {
      const response = await axios.patch(
        `/api/updateTeacher/${data.UserID}`,
        data
      );

      if (response.data.success) {
        toast.success(response.data.message); // Show success message
        // Optionally refetch the updated data
        const updatedResponse = await axios.get(
          `/api/getTeacher/${data.UserID}`
        );
        setData(updatedResponse.data); // Update state with the new data
      } else {
        // Handle known errors returned from the server
        toast.error(response.data.message || "Update failed.");
      }
    } catch (error) {
      // Handle unexpected errors
      console.error(error);
      toast.error("An unexpected error occurred. Please try again later.");
    }
  };

  return (
    <div className="p-10">
      <div className="flex items-center justify-start gap-2 mb-5">
        <h1 className="text-3xl font-semibold">Your Profile</h1>
        <Tooltip
          showArrow={true}
          content={
            <div className="p-2">
              <div className="text-sm font-bold">Profile Information</div>
              <div className="text-xs">
                This function will view your information.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
        </Tooltip>
      </div>
      <div className="w-full max-w-8xl mx-auto bg-white shadow-lg rounded-lg p-[3rem]">
        {/* Profile Header */}
        <div className="flex flex-col md:flex-row items-center md:items-start gap-6 relative">
          <Tooltip
            content={
              <div className="p-2">
                <div className="text-sm font-bold">Profile Picture Update</div>
                <div className="text-xs">
                  You can update your profile picture once clicked.
                </div>
              </div>
            }
          >
            <label className="cursor-pointer">
              <img
                src={profileImg}
                className="w-32 h-32 rounded-full"
                alt="Profile"
              />
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>
          </Tooltip>
          <div className="flex-1">
            <h2 className="text-2xl font-bold">
              {data ? `${data.FirstName} ${data.LastName}` : "Teacher Name"}
            </h2>
            <h3>
              {user ? `Adviser of ${data.Section}` : "Adviser of Section"}
            </h3>
            <div className="flex flex-col mt-4 text-gray-700">
              <div className="flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faAddressCard}
                  className="text-gray-600"
                />
                <span>
                  Teacher ID Number: {data ? data.UserID : "Teacher ID"}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <FontAwesomeIcon icon={faEnvelope} className="text-gray-600" />
                <span>
                  Email Address: {data ? data.email : "Email Address"}
                </span>
              </div>
            </div>
          </div>
          <Button
            color="primary"
            className="w-[120px] text-md p-1"
            onClick={() => navigate(-1)}
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
              Update
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
                    Civil Status:
                  </span>
                  <p className="text-gray-800">{data.Status || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Last Name:
                  </span>
                  <p className="text-gray-800">{data.LastName || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Gender:</span>
                  <p className="text-gray-800">{data.Gender || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Age:</span>
                  <p className="text-gray-800">{data.Age || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Birthday:</span>
                  <p className="text-gray-800">{data.Birthday || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Address:</span>
                  <p className="text-gray-800">{data.Address || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Contact Number:
                  </span>
                  <p className="text-gray-800">{data.ContactNumber || "N/A"}</p>
                </div>
              </div>
            </div>
          )}
          {activeIndex === 1 && (
            <div className="mt-6 bg-[#faf9f4] p-6 rounded-lg shadow">
              {/* Form for updating profile */}
              <form onSubmit={handleUpdate}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Input
                    label="First Name"
                    value={data.FirstName}
                    onChange={(e) =>
                      setData({ ...data, FirstName: e.target.value })
                    }
                  />
                  <Input
                    label="Last Name"
                    value={data.LastName}
                    onChange={(e) =>
                      setData({ ...data, LastName: e.target.value })
                    }
                  />
                  <Input
                    label="Gender"
                    value={data.Gender}
                    onChange={(e) =>
                      setData({ ...data, Gender: e.target.value })
                    }
                  />
                  <Select
                    label="Civil Status"
                    value={data.Status}
                    onChange={(e) =>
                      setData({ ...data, Status: e.target.value })
                    }
                  >
                    <SelectItem value="Single">Single</SelectItem>
                    <SelectItem value="Married">Married</SelectItem>
                    <SelectItem value="Divorced">Divorced</SelectItem>
                  </Select>
                  <Input
                    label="Birthday"
                    type="date"
                    value={data.Birthday}
                    onChange={(e) =>
                      setData({ ...data, Birthday: e.target.value })
                    }
                  />
                  <Input
                    label="Address"
                    value={data.Address}
                    onChange={(e) =>
                      setData({ ...data, Address: e.target.value })
                    }
                  />
                  <Input
                    label="Contact Number"
                    value={data.ContactNumber}
                    onChange={(e) =>
                      setData({ ...data, ContactNumber: e.target.value })
                    }
                  />
                  <Input
                    label="Section"
                    value={data.Section}
                    onChange={(e) =>
                      setData({ ...data, Section: e.target.value })
                    }
                  />
                </div>
                <div className="mt-4 flex justify-end">
                  <Button type="submit" color="primary">
                    Update
                  </Button>
                </div>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BodyTeacherProfileBasic;
