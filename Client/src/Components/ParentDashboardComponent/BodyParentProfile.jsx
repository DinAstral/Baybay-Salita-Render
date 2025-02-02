import React, { useState, useEffect, useContext } from "react";
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

const BodyParentProfile = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useContext(UserContext);
  const [activeIndex, setActiveIndex] = useState(0);
  const [modalShow, setModalShow] = useState(false);
  const [profileImg, setProfileImg] = useState(profileImage);
  const [parent, setParent] = useState({});
  const [age, setAge] = useState(null);
  const [data, setData] = useState({});

  useEffect(() => {
    axios
      .get(`/api/getParent/${user.UserID}`)
      .then((response) => {
        setProfileImg(response.data.Picture || profileImage);
        setParent(response.data);
        setData(response.data); // Initialize the data state with the parent's information
      })
      .catch(() => {
        toast.error("Failed to fetch parent data. Please try again later.");
      });
  }, [user.UserID]);

  useEffect(() => {
    const menuItem = ["/parentProfile"];
    const index = menuItem.findIndex((item) => item === location.pathname);
    setActiveIndex(index);
  }, [location.pathname]);

  useEffect(() => {
    if (parent.Birthday) {
      const calculateAge = (birthday) => {
        const birthDate = new Date(birthday);
        const today = new Date();
        let age = today.getFullYear() - birthDate.getFullYear();
        const monthDiff = today.getMonth() - birthDate.getMonth();
        if (
          monthDiff < 0 ||
          (monthDiff === 0 && today.getDate() < birthDate.getDate())
        ) {
          age--;
        }
        return age;
      };
      setAge(calculateAge(parent.Birthday));
    }
  }, [parent.Birthday]);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append("Profile", file);
      formData.append("role", user.role); // Adjust role according to your user context
      formData.append("UserID", user.UserID);

      try {
        const response = await axios.post("/api/profileUpdate", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        if (response.data.message) {
          setProfileImg(response.data.updatedProfileUrl); // Update profile image URL
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

  const editParent = async (event) => {
    event.preventDefault(); // Prevent the default form submission

    try {
      const response = await axios.patch(
        `/api/updateParent/${data.UserID}`,
        data
      );

      // Check if the response indicates success
      if (response.status === 200 && response.data) {
        toast.success("Your profile updated successfully!");

        // Optionally, you can refetch the data after the update
        const updatedResponse = await axios.get(
          `/api/getParent/${data.UserID}`
        );
        setData(updatedResponse.data);
      } else {
        toast.error(`Error: ${response.data.error || "Update failed."}`);
      }
    } catch (error) {
      console.error("Error updating parent profile:", error); // Log the error for debugging

      // Check if error response from server
      if (error.response) {
        // Server responded with a status other than 2xx
        toast.error(
          `Failed to update parent profile: ${
            error.response.data.error || "Please try again later."
          }`
        );
      } else if (error.request) {
        // Request was made but no response received
        toast.error("No response from server. Please check your connection.");
      } else {
        // Something else caused the error
        toast.error("Failed to update parent profile. Please try again later.");
      }
    }
  };

  return (
    <div className="p-10">
      <div className="flex items-center justify-start gap-2 mb-5">
        <h1 className="text-3xl font-semibold">Parent Profile</h1>
        <Tooltip
          showArrow={true}
          content={
            <div className="p-2">
              <div className="text-sm font-bold">Profile Information</div>
              <div className="text-xs">
                This function will view your profile.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
        </Tooltip>
      </div>

      <div className="w-full max-w-8xl mx-auto bg-white shadow-lg rounded-lg p-[3rem]">
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
              {user ? `${user.FirstName} ${user.LastName}` : "Parent Name"}
            </h2>
            <h3>{user ? user.role : "Role"}</h3>

            <div className="flex flex-col mt-4 text-gray-700">
              <div className="flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faAddressCard}
                  className="text-gray-600"
                />
                <span>
                  Parent ID Number: {user ? user.UserID : "Parent ID"}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <FontAwesomeIcon icon={faEnvelope} className="text-gray-600" />
                <span>
                  Email Address: {user ? user.email : "Email Address"}
                </span>
              </div>
            </div>
          </div>

          <Button color="primary" onClick={() => navigate(-1)}>
            Back
          </Button>
        </div>

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

          {activeIndex === 0 && (
            <div className="mt-6 bg-[#faf9f4] p-6 rounded-lg shadow">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <span className="block text-sm text-gray-600">
                    First Name:
                  </span>
                  <p className="text-gray-800">
                    {user ? user.FirstName : "N/A"}
                  </p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Contact Number:
                  </span>
                  <p className="text-gray-800">
                    {parent.ContactNumber || "N/A"}
                  </p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">
                    Last Name:
                  </span>
                  <p className="text-gray-800">
                    {user ? user.LastName : "N/A"}
                  </p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Gender:</span>
                  <p className="text-gray-800">{parent.Gender || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Birthday:</span>
                  <p className="text-gray-800">{parent.Birthday || "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Age:</span>
                  <p className="text-gray-800">{age !== null ? age : "N/A"}</p>
                </div>
                <div>
                  <span className="block text-sm text-gray-600">Address:</span>
                  <p className="text-gray-800">{parent.Address || "N/A"}</p>
                </div>
              </div>
            </div>
          )}

          {activeIndex === 1 && (
            <form
              onSubmit={editParent}
              className="mt-6 bg-[#faf9f4] p-6 rounded-lg shadow"
            >
              <h2 className="text-lg font-semibold mb-4">
                Update Your Profile
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  required
                  label="First Name"
                  value={data.FirstName || ""}
                  onChange={(e) =>
                    setData({ ...data, FirstName: e.target.value })
                  }
                />
                <Input
                  required
                  label="Last Name"
                  value={data.LastName || ""}
                  onChange={(e) =>
                    setData({ ...data, LastName: e.target.value })
                  }
                />
                <Select
                  label="Gender"
                  value={data.Gender || ""}
                  onChange={(e) => setData({ ...data, Gender: e })}
                >
                  <SelectItem value="Male">Male</SelectItem>
                  <SelectItem value="Female">Female</SelectItem>
                  <SelectItem value="Other">Other</SelectItem>
                </Select>
                <Input
                  required
                  type="date"
                  label="Birthday"
                  value={data.Birthday || ""}
                  onChange={(e) =>
                    setData({ ...data, Birthday: e.target.value })
                  }
                />
                <Input
                  required
                  label="Contact Number"
                  value={data.ContactNumber || ""}
                  onChange={(e) =>
                    setData({ ...data, ContactNumber: e.target.value })
                  }
                />
                <Input
                  label="Address"
                  value={data.Address || ""}
                  onChange={(e) =>
                    setData({ ...data, Address: e.target.value })
                  }
                />
              </div>
              <div className="mt-4">
                <Button type="submit" color="primary">
                  Update Profile
                </Button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default BodyParentProfile;
