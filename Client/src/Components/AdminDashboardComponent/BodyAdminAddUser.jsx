import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

function generateRandomCode(role, length) {
  const characters = "0123456789";
  let prefix = "";
  switch (role) {
    case "Teacher":
      prefix = "teacherID_";
      break;
    case "Admin":
      prefix = "adminID_";
      break;
    default:
      break;
  }
  let result = prefix;
  const charactersLength = characters.length;
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

const BodyAdminAddUser = () => {
  const navigate = useNavigate();

  const [data, setData] = useState({
    UserID: "",
    FirstName: "",
    LastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    role: "",
  });

  const [errors, setErrors] = useState({}); // Initialize errors state

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

    if (errors.length > 0) {
      return errors.join(" ");
    }

    return null; // No errors
  }

  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    setErrors({}); // Clear previous errors

    if (!data.FirstName) {
      newErrors.FirstName = "First Name is required.";
      isValid = false;
    }
    if (!data.LastName) {
      newErrors.LastName = "Last Name is required.";
      isValid = false;
    }
    if (!data.email) {
      newErrors.email = "Email is required.";
      isValid = false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(data.email)) {
      newErrors.email = "Invalid email format.";
      isValid = false;
    }

    const passwordError = validatePassword(data.password);
    if (passwordError) {
      newErrors.password = passwordError;
      isValid = false;
    }

    if (data.password !== data.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match.";
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setData((prevData) => ({ ...prevData, [name]: value }));
  };

  const addUser = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out the form correctly.");
      return;
    }

    const UserID = generateRandomCode(data.role, 6);

    try {
      const response = await axios.post("/api/addUser", {
        ...data,
        UserID,
      });

      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({
          UserID: "",
          FirstName: "",
          LastName: "",
          email: "",
          password: "",
          confirmPassword: "",
          role: "",
        });
        toast.success("User added successfully.", { duration: 4000 });
        navigate("/adminUsers");
      }
    } catch (error) {
      console.error("There was an error adding the user:", error);
      toast.error(
        error.response?.data?.message ||
          "Failed to add user. Please try again later."
      );
    }
  };

  return (
    <div className="container mx-auto px-6 pt-[4rem]">
      <div className="bg-white shadow-md rounded-md px-8 pt-6 pb-8 mb-4 flex flex-col">
        <h2 className="mb-6 text-3xl font-bold text-gray-700 flex items-center gap-2">
          Add User's Information
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">Add User</div>
                <div className="text-tiny">
                  This function will add the information of the user to the
                  system.
                </div>
              </div>
            }
          >
            <FontAwesomeIcon
              icon={faCircleInfo}
              size="sm"
              className="text-gray-700 text-[20px]"
            />
          </Tooltip>
        </h2>
        <form onSubmit={addUser}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input
              label="First Name"
              type="text"
              name="FirstName"
              placeholder="Enter First Name"
              value={data.FirstName}
              errorMessage={errors.FirstName}
              isInvalid={!!errors.FirstName}
              onChange={handleChange}
            />
            <Input
              label="Last Name"
              type="text"
              name="LastName"
              placeholder="Enter Last Name"
              value={data.LastName}
              errorMessage={errors.LastName}
              isInvalid={!!errors.LastName}
              onChange={handleChange}
            />
            <Input
              label="Email"
              type="email"
              name="email"
              placeholder="Enter Email"
              value={data.email}
              errorMessage={errors.email}
              isInvalid={!!errors.email}
              onChange={handleChange}
            />
            <Input
              label="Password"
              type="password"
              name="password"
              placeholder="Set Password"
              value={data.password}
              errorMessage={errors.password}
              isInvalid={!!errors.password}
              onChange={handleChange}
            />
            <Input
              label="Confirm Password"
              type="password"
              name="confirmPassword"
              placeholder="Confirm Password"
              value={data.confirmPassword}
              errorMessage={errors.confirmPassword}
              isInvalid={!!errors.confirmPassword}
              onChange={handleChange}
            />
            <Select
              label="Role"
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
          <div className="mt-6">
            <Button type="submit" color="primary" size="lg" radius="sm">
              Add User
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BodyAdminAddUser;
