/* eslint-disable react/no-unescaped-entities */
import { useState, useEffect } from "react";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import { useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

function validatePassword(password) {
  const minLength = 8;
  const errors = [];

  const hasUpperCase = /[A-Z]/.test(password);
  const hasLowerCase = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

  if (!password) {
    return "Password is required.";
  }

  if (password.length < minLength) {
    errors.push("Password must be at least 8 characters long.");
  }
  if (!hasUpperCase) {
    errors.push("Password must contain at least one uppercase letter.");
  }
  if (!hasLowerCase) {
    errors.push("Password must contain at least one lowercase letter.");
  }
  if (!hasDigit) {
    errors.push("Password must contain at least one digit.");
  }
  if (!hasSpecialChar) {
    errors.push("Password must contain at least one special character.");
  }

  return errors.length > 0 ? errors.join(" ") : null;
}

const BodyAdminEditUser = () => {
  const navigate = useNavigate();
  const { UserID } = useParams(); // Get UserID from URL parameters

  const [data, setData] = useState({
    FirstName: "",
    LastName: "",
    email: "",
    password: "",
    role: "",
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`/api/getUserID/${UserID}`); // Use UserID to fetch user data
        setData((prevData) => ({
          ...prevData,
          ...response.data,
          password: "", // Reset password to empty when loading data
        }));
      } catch (err) {
        console.error("Error fetching user data:", err);
        toast.error("Failed to fetch user data. Please try again later.");
      }
    };
    fetchData();
  }, [UserID]);

  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    // First Name validation
    if (!data.FirstName) {
      newErrors.FirstName = "First Name is required.";
      isValid = false;
    }

    // Last Name validation
    if (!data.LastName) {
      newErrors.LastName = "Last Name is required.";
      isValid = false;
    }

    // Email validation
    if (!data.email) {
      newErrors.email = "Email is required.";
      isValid = false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(data.email)) {
      newErrors.email = "Invalid email format.";
      isValid = false;
    }

    // Password validation
    if (data.password) {
      const passwordError = validatePassword(data.password);
      if (passwordError) {
        newErrors.password = passwordError;
        isValid = false;
      }
    }

    setErrors(newErrors);
    return isValid;
  };

  const editUser = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out the form correctly.");
      return;
    }

    try {
      const { FirstName, LastName, email, password, role } = data;
      const response = await axios.patch(`/api/updateUser/${UserID}`, {
        // Pass UserID in the patch request
        FirstName,
        LastName,
        email,
        password,
        role,
      });

      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        toast.success("Updated User info successfully.");
        navigate("/AdminUsers");
      }
    } catch (error) {
      console.error("Error updating user data:", error);
      toast.error("Failed to update user. Please try again.");
    }
  };

  return (
    <div className="container mx-auto px-6 pt-[4rem]">
      <div className="bg-white shadow-md rounded-md px-8 pt-6 pb-8 mb-4 flex flex-col">
        <div className="flex items-center gap-3 mb-6">
          <h2
            id="update-user-info"
            className="text-2xl font-bold text-gray-800"
          >
            Update User's Information
          </h2>
          <Tooltip
            aria-labelledby="update-user-info"
            content={
              <div className="px-1 py-2">
                <div className="text-sm font-bold">Update Information</div>
                <div className="text-xs">
                  This function will update the information of the user in the
                  system.
                </div>
              </div>
            }
          >
            <FontAwesomeIcon
              icon={faCircleInfo}
              size="lg"
              className="text-gray-600"
              aria-label="Information icon"
            />
          </Tooltip>
        </div>
        <form onSubmit={editUser}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* First Name */}
            <div className="flex flex-col">
              <label htmlFor="first-name" className="font-medium text-gray-700">
                First Name
              </label>
              <Input
                id="first-name"
                aria-label="First Name"
                underlined
                type="text"
                placeholder="Enter First Name"
                value={data.FirstName}
                errorMessage={errors.FirstName}
                isInvalid={!!errors.FirstName}
                onChange={(e) =>
                  setData({ ...data, FirstName: e.target.value })
                }
              />
            </div>

            {/* Last Name */}
            <div className="flex flex-col">
              <label htmlFor="last-name" className="font-medium text-gray-700">
                Last Name
              </label>
              <Input
                id="last-name"
                aria-label="Last Name"
                underlined
                type="text"
                placeholder="Enter Last Name"
                value={data.LastName}
                errorMessage={errors.LastName}
                isInvalid={!!errors.LastName}
                onChange={(e) => setData({ ...data, LastName: e.target.value })}
              />
            </div>

            {/* Email */}
            <div className="flex flex-col">
              <label htmlFor="email" className="font-medium text-gray-700">
                Email
              </label>
              <Input
                id="email"
                aria-label="Email"
                underlined
                type="email"
                placeholder="Enter the User's Email"
                value={data.email}
                errorMessage={errors.email}
                isInvalid={!!errors.email}
                onChange={(e) => setData({ ...data, email: e.target.value })}
              />
            </div>

            {/* Password */}
            <div className="flex flex-col">
              <label htmlFor="password" className="font-medium text-gray-700">
                Password
              </label>
              <Input
                id="password"
                aria-label="Password"
                underlined
                type="password"
                placeholder="Set Password"
                value={data.password}
                errorMessage={errors.password}
                isInvalid={!!errors.password}
                onChange={(e) => setData({ ...data, password: e.target.value })}
              />
            </div>

            {/* Role */}
            <div className="flex flex-col">
              <label htmlFor="role" className="font-medium text-gray-700">
                Role
              </label>
              <Select
                id="role"
                aria-label="Role"
                placeholder="Select Role"
                value={data.role}
                errorMessage={errors.role}
                isInvalid={!!errors.role}
                onChange={(e) => setData({ ...data, role: e.target.value })}
              >
                <SelectItem key="Admin" value="Admin">
                  Admin
                </SelectItem>
                <SelectItem key="Teacher" value="Teacher">
                  Teacher
                </SelectItem>
              </Select>
            </div>
          </div>
          <div className="flex justify-end mt-6">
            <Button type="submit" color="primary" aria-label="Save Changes">
              Save Changes
            </Button>
            <Button
              className="bg-[#ff505b] text-white ml-4"
              onClick={() => navigate(-1)}
              aria-label="Cancel"
            >
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BodyAdminEditUser;
