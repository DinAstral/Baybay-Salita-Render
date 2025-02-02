import { useState } from "react";
import axios from "axios";
import { toast } from "react-hot-toast";
import {
  Input,
  Button,
  Card,
  CardBody,
  Select,
  SelectItem,
} from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useNavigate } from "react-router-dom";
import {
  faUser,
  faEnvelope,
  faEye,
  faEyeSlash,
  faPhone,
  faLocationDot,
} from "@fortawesome/free-solid-svg-icons";

const TeacherRegister = () => {
  const [data, setData] = useState({
    FirstName: "",
    LastName: "",
    Section: "",
    Department: "",
    Birthday: "",
    Age: "",
    Gender: "",
    Address: "",
    Status: "",
    ContactNumber: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [errors, setErrors] = useState({});
  const navigate = useNavigate();

  // State to manage tabs
  const [activeIndex, setActiveIndex] = useState(0);

  const [isVisible, setIsVisible] = useState(false);
  const toggleVisibility = () => setIsVisible(!isVisible);

  const [isVisibleConfirm, setIsVisibleConfirm] = useState(false);
  const toggleVisibilityConfirm = () => setIsVisibleConfirm(!isVisibleConfirm);

  const registerButton = () => {
    navigate("/register");
  };

  // Helper function to calculate age based on birthday
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

  function validatePassword(password) {
    const minLength = 8;
    const errors = [];

    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasDigit = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    // Check if password is provided
    if (!password) {
      return "Password is required.";
    }

    // Validate the different criteria and accumulate errors
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

    // If there are any errors, return them as a combined string
    if (errors.length > 0) {
      return errors.join(" ");
    }

    return null; // No errors
  }

  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    // Clear previous errors before validation
    setErrors({});

    if (!data.FirstName) {
      newErrors.FirstName = "First Name is required.";
      isValid = false;
    }
    if (!data.LastName) {
      newErrors.LastName = "Last Name is required.";
      isValid = false;
    }
    if (!data.Section) {
      newErrors.Section = "Section is required.";
      isValid = false;
    }
    if (!data.Department) {
      newErrors.Department = "Department is required.";
      isValid = false;
    }
    if (!data.Birthday) {
      newErrors.Birthday = "Birthday is required.";
      isValid = false;
    }
    // Age validation: must be >= 5 years old
    if (data.Age < 20) {
      newErrors.Birthday = "Age must be at least 20 years old.";
      isValid = false;
    }
    if (!data.Gender) {
      newErrors.Gender = "Gender is required.";
      isValid = false;
    }
    if (!data.Address) {
      newErrors.Address = "Address is required.";
      isValid = false;
    }
    if (!data.Status) {
      newErrors.Status = "Status is required.";
      isValid = false;
    }
    if (!data.ContactNumber) {
      newErrors.ContactNumber = "Contact Number is required.";
      isValid = false;
    }
    const phoneRegex = /^09\d{9}$/;
    if (!phoneRegex.test(data.ContactNumber)) {
      newErrors.ContactNumber = "Please enter a valid PH contact number.";
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
    } else if (data.password !== data.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match.";
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };
  const registerTeacher = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out this form.");
      return;
    }

    const {
      FirstName,
      LastName,
      Section,
      Department,
      Birthday,
      Address,
      Status,
      Gender,
      ContactNumber,
      email,
      password,
      Age,
    } = data;

    try {
      const response = await axios.post("/api/registerTeacher", {
        FirstName,
        LastName,
        Section,
        Department,
        Birthday,
        Age,
        Address,
        Status,
        Gender,
        ContactNumber,
        email,
        password,
      });

      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({});
        toast.success(response.data.message);
        localStorage.setItem("userId", response.data.data.userId);
        navigate("/verifyEmail");
      }
    } catch (error) {
      toast.error("An error occurred. Please try again.");
    }
  };

  return (
    <div className="flex bg-[#f6fbff] w-full items-center min-h-screen p-4 md:p-8">
      <div className="w-full max-w-4xl mx-auto">
        <Card className="w-full flex flex-col p-4">
          <h1 className="text-2xl md:text-3xl font-bold text-center">
            Teacher Registration
          </h1>
          {/* Tabs Header */}
          <div className="flex gap-4 md:gap-6 border-b pb-3 justify-center mb-4">
            <Button
              variant="light"
              className={`text-sm md:text-md font-medium ${
                activeIndex === 0 ? "border-b-4 border-blue-500" : ""
              }`}
              onClick={() => setActiveIndex(0)}
            >
              Teacher Information
            </Button>
          </div>
          <p className="text-sm text-center mb-4">
            Please fill up the details needed!
          </p>
          <CardBody>
            {activeIndex === 0 && (
              <form onSubmit={registerTeacher}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* First Name, Last Name */}
                  <Input
                    name="FirstName"
                    label="First Name"
                    variant="bordered"
                    value={data.FirstName}
                    onChange={(e) =>
                      setData({ ...data, FirstName: e.target.value })
                    }
                    errorMessage={errors.FirstName}
                    isInvalid={!!errors.FirstName}
                    endContent={<FontAwesomeIcon icon={faUser} />}
                  />
                  <Input
                    name="LastName"
                    label="Last Name"
                    variant="bordered"
                    value={data.LastName}
                    onChange={(e) =>
                      setData({ ...data, LastName: e.target.value })
                    }
                    errorMessage={errors.LastName}
                    isInvalid={!!errors.LastName}
                    endContent={<FontAwesomeIcon icon={faUser} />}
                  />

                  {/* Section, Department */}
                  <Select
                    label="Section"
                    placeholder="Select your section"
                    variant="bordered"
                    value={data.Section}
                    onChange={(e) =>
                      setData({ ...data, Section: e.target.value })
                    }
                    errorMessage={errors.Section}
                    isInvalid={!!errors.Section}
                  >
                    <SelectItem key="Aster">Aster</SelectItem>
                    <SelectItem key="Camia">Camia</SelectItem>
                    <SelectItem key="Dahlia">Dahlia</SelectItem>
                    <SelectItem key="Iris">Iris</SelectItem>
                    <SelectItem key="Jasmin">Jasmin</SelectItem>
                    <SelectItem key="Orchid">Orchid</SelectItem>
                    <SelectItem key="Rose">Rose</SelectItem>
                    <SelectItem key="Tulip">Tulip</SelectItem>
                    <SelectItem key="SSC">SSC</SelectItem>
                  </Select>
                  <Select
                    label="Department"
                    placeholder="Select your department"
                    variant="bordered"
                    value={data.Department}
                    onChange={(e) =>
                      setData({ ...data, Department: e.target.value })
                    }
                    errorMessage={errors.Department}
                    isInvalid={!!errors.Department}
                  >
                    <SelectItem key="Filipino">Filipino</SelectItem>
                  </Select>

                  {/* Birthday, Gender */}
                  <Input
                    name="Birthday"
                    type="date"
                    label="Birthday"
                    variant="bordered"
                    value={data.Birthday}
                    onChange={(e) => {
                      const birthday = e.target.value;
                      const age = calculateAge(birthday);
                      setData({ ...data, Birthday: birthday, Age: age });
                    }}
                    errorMessage={errors.Birthday}
                    isInvalid={!!errors.Birthday}
                  />
                  <Select
                    label="Gender"
                    placeholder="Select your gender"
                    variant="bordered"
                    value={data.Gender}
                    onChange={(e) =>
                      setData({ ...data, Gender: e.target.value })
                    }
                    errorMessage={errors.Gender}
                    isInvalid={!!errors.Gender}
                  >
                    <SelectItem key="Male">Male</SelectItem>
                    <SelectItem key="Female">Female</SelectItem>
                  </Select>

                  {/* Address, Status */}
                  <Input
                    name="Address"
                    label="Address"
                    variant="bordered"
                    value={data.Address}
                    onChange={(e) =>
                      setData({ ...data, Address: e.target.value })
                    }
                    errorMessage={errors.Address}
                    isInvalid={!!errors.Address}
                    endContent={<FontAwesomeIcon icon={faLocationDot} />}
                  />
                  <Select
                    name="Status"
                    label="Status"
                    variant="bordered"
                    placeholder="Select your status"
                    value={data.Status}
                    onChange={(e) =>
                      setData({ ...data, Status: e.target.value })
                    }
                    errorMessage={errors.Status}
                    isInvalid={!!errors.Status}
                  >
                    <SelectItem key="Single">Single</SelectItem>
                  </Select>

                  {/* Contact Number, Email */}
                  <Input
                    name="ContactNumber"
                    label="Contact Number"
                    maxLength={11}
                    variant="bordered"
                    value={data.ContactNumber}
                    onChange={(e) =>
                      setData({ ...data, ContactNumber: e.target.value })
                    }
                    errorMessage={errors.ContactNumber}
                    isInvalid={!!errors.ContactNumber}
                    endContent={<FontAwesomeIcon icon={faPhone} />}
                  />
                  <Input
                    name="email"
                    label="Email"
                    variant="bordered"
                    value={data.email}
                    onChange={(e) =>
                      setData({ ...data, email: e.target.value })
                    }
                    errorMessage={errors.email}
                    isInvalid={!!errors.email}
                    endContent={<FontAwesomeIcon icon={faEnvelope} />}
                  />
                </div>

                {/* Password and Confirm Password */}
                <div className="mt-3">
                  <Input
                    name="password"
                    label="Password"
                    type={isVisible ? "text" : "password"}
                    variant="bordered"
                    value={data.password}
                    onChange={(e) =>
                      setData({ ...data, password: e.target.value })
                    }
                    errorMessage={errors.password}
                    isInvalid={!!errors.password}
                    endContent={
                      <button type="button" onClick={toggleVisibility}>
                        <FontAwesomeIcon
                          icon={isVisible ? faEyeSlash : faEye}
                        />
                      </button>
                    }
                  />
                  <Input
                    name="confirmPassword"
                    label="Confirm Password"
                    type={isVisibleConfirm ? "text" : "password"}
                    variant="bordered"
                    value={data.confirmPassword}
                    className="mt-3"
                    onChange={(e) =>
                      setData({ ...data, confirmPassword: e.target.value })
                    }
                    errorMessage={errors.confirmPassword}
                    isInvalid={!!errors.confirmPassword}
                    endContent={
                      <button type="button" onClick={toggleVisibilityConfirm}>
                        <FontAwesomeIcon
                          icon={isVisibleConfirm ? faEyeSlash : faEye}
                        />
                      </button>
                    }
                  />
                </div>

                {/* Submit and Cancel Buttons */}
                <div className="flex flex-col md:flex-row items-center justify-center gap-4 mt-6">
                  <Button
                    className="w-full md:w-1/2 text-md"
                    color="danger"
                    variant="light"
                    onClick={registerButton}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    className="w-full md:w-1/2 text-md"
                    color="primary"
                  >
                    Submit
                  </Button>
                </div>
              </form>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
};

export default TeacherRegister;
