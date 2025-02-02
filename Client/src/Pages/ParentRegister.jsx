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
  CardFooter,
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
  faIdCard,
} from "@fortawesome/free-solid-svg-icons";

const ParentRegister = () => {
  const navigate = useNavigate();
  const [activeIndex, setActiveIndex] = useState(0); // Tab management
  const [isVisible, setIsVisible] = useState(false);
  const [isVisibleConfirm, setIsVisibleConfirm] = useState(false);
  const [errors, setErrors] = useState({});

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

  const [data, setData] = useState({
    FirstName: "",
    LastName: "",
    StudentFirstName: "",
    StudentLastName: "",
    LRN: "",
    Birthday: "",
    Age: "",
    StudentBirthday: "",
    StudentAge: "",
    Gender: "",
    StudentGender: "",
    Address: "",
    Status: "",
    ContactNumber: "",
    MotherTongue: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const toggleVisibility = () => setIsVisible(!isVisible);
  const toggleVisibilityConfirm = () => setIsVisibleConfirm(!isVisibleConfirm);

  // Helper function to check if input is a valid number
  const isNumber = (input) => !isNaN(input);

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

    setErrors({});

    if (!data.FirstName) {
      newErrors.FirstName = "First Name is required.";
      isValid = false;
    }
    if (!data.LastName) {
      newErrors.LastName = "Last Name is required.";
      isValid = false;
    }
    if (!data.LRN) {
      newErrors.LRN = "LRN is required.";
      isValid = false;
    }
    // Validate LRN
    if (!isNumber(data.LRN)) {
      newErrors.LRN = "Invalid LRN.";
      isValid = false;
    }
    if (!data.StudentFirstName) {
      newErrors.StudentFirstName = "Student First Name is required.";
      isValid = false;
    }
    if (!data.StudentLastName) {
      newErrors.StudentLastName = "Student Last Name is required.";
      isValid = false;
    }
    if (!data.Birthday) {
      newErrors.Birthday = "Birthday is required.";
      isValid = false;
    } else {
      const age = calculateAge(data.Birthday);
      if (age < 18) {
        newErrors.Birthday = "Parent must be at least 18 years old.";
        isValid = false;
      }
    }

    if (!data.StudentBirthday) {
      newErrors.StudentBirthday = "Student Birthday is required.";
      isValid = false;
    } else {
      const studentAge = calculateAge(data.StudentBirthday);
      if (studentAge < 5) {
        newErrors.StudentBirthday = "Student must be at least 5 years old.";
        isValid = false;
      }
    }

    if (!data.MotherTongue) {
      newErrors.MotherTongue = "MotherTongue is required.";
      isValid = false;
    }
    if (!data.Gender) {
      newErrors.Gender = "Gender is required.";
      isValid = false;
    }
    if (!data.StudentGender) {
      newErrors.StudentGender = "Student Gender is required.";
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
    if (!isNumber(data.ContactNumber)) {
      newErrors.ContactNumber = "Invalid Contact Number.";
      isValid = false;
    }
    // Validate Contact Number (starts with 09 and 11 digits long)
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
    // Validate email format
    if (!emailRegex.test(data.email)) {
      newErrors.email = "Invalid email format.";
    }

    const passwordError = validatePassword(data.password);
    if (passwordError) {
      newErrors.password = passwordError;
      isValid = false;
    }

    // Validate passwords match
    else if (data.password !== data.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match.";
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const registerParent = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out this form.");
      return;
    }

    const {
      FirstName,
      LastName,
      Birthday,
      Age,
      Address,
      Status,
      Gender,
      ContactNumber,
      email,
      password,
      StudentFirstName,
      StudentLastName,
      LRN,
      StudentBirthday,
      StudentAge,
      StudentGender,
      MotherTongue,
    } = data;

    try {
      const response = await axios.post("/api/registerParent", {
        FirstName,
        LastName,
        Birthday,
        Age,
        Address,
        Status,
        Gender,
        ContactNumber,
        email,
        password,
        StudentFirstName,
        StudentLastName,
        LRN,
        StudentBirthday,
        StudentAge,
        StudentGender,
        MotherTongue,
      });

      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({});
        toast.success("Register Verification.");
        localStorage.setItem("userId", response.data.data.userId);
        navigate("/verifyEmail");
      }
    } catch (error) {
      console.log(error);
      toast.error("An error occurred. Please try again.");
    }
  };

  return (
    <div className="p-4 md:p-10 bg-[#f6fbff] flex w-full min-h-screen items-center">
      <div className="w-full max-w-4xl mx-auto bg-white shadow-lg rounded-lg p-6 md:p-10">
        <h1 className="text-2xl md:text-3xl font-semibold text-center mb-6">
          Parent Registration
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
            Parent Information
          </Button>
          <Button
            variant="light"
            className={`text-sm md:text-md font-medium ${
              activeIndex === 1 ? "border-b-4 border-blue-500" : ""
            }`}
            onClick={() => setActiveIndex(1)}
          >
            Student Information
          </Button>
        </div>

        {/* Tabs Content */}
        <Card>
          <form onSubmit={registerParent}>
            <CardBody>
              <div
                className={`${
                  activeIndex === 0 ? "block" : "hidden"
                } grid grid-cols-1 md:grid-cols-2 gap-4`}
              >
                <Input
                  type="text"
                  name="FirstName"
                  label="First Name"
                  variant="bordered"
                  value={data.FirstName}
                  onChange={(e) =>
                    setData({ ...data, FirstName: e.target.value })
                  }
                  errorMessage={errors.FirstName}
                  isInvalid={!!errors.FirstName}
                  endContent={
                    <FontAwesomeIcon
                      icon={faUser}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Input
                  type="text"
                  name="LastName"
                  label="Last Name"
                  variant="bordered"
                  value={data.LastName}
                  onChange={(e) =>
                    setData({ ...data, LastName: e.target.value })
                  }
                  errorMessage={errors.LastName}
                  isInvalid={!!errors.LastName}
                  endContent={
                    <FontAwesomeIcon
                      icon={faUser}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Input
                  name="Birthday"
                  type="date"
                  label="Birthday"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.Birthday}
                  onChange={(e) => {
                    const birthday = e.target.value;
                    const age = calculateAge(birthday);
                    setData({ ...data, Birthday: birthday, Age: age });
                  }}
                  errorMessage={errors.Birthday}
                  isInvalid={!!errors.Birthday}
                />
                <Input
                  type="text"
                  name="ContactNumber"
                  label="Contact Number"
                  maxLength={11}
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.ContactNumber}
                  onChange={(e) =>
                    setData({ ...data, ContactNumber: e.target.value })
                  }
                  errorMessage={errors.ContactNumber}
                  isInvalid={!!errors.ContactNumber}
                  endContent={
                    <FontAwesomeIcon
                      icon={faPhone}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Select
                  label="Gender"
                  name="Gender"
                  placeholder="Select your gender"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.Gender || ""}
                  onChange={(e) => setData({ ...data, Gender: e.target.value })}
                  errorMessage={errors.Gender}
                  isInvalid={!!errors.Gender}
                >
                  <SelectItem key="Male">Male</SelectItem>
                  <SelectItem key="Female">Female</SelectItem>
                  <SelectItem key="Other">Other</SelectItem>
                </Select>
                <Input
                  type="text"
                  name="Address"
                  label="Address"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.Address}
                  onChange={(e) =>
                    setData({ ...data, Address: e.target.value })
                  }
                  errorMessage={errors.Address}
                  isInvalid={!!errors.Address}
                  endContent={
                    <FontAwesomeIcon
                      icon={faLocationDot}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Select
                  label="Status"
                  name="Status"
                  placeholder="Select your status"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.Status}
                  onChange={(e) => setData({ ...data, Status: e.target.value })}
                  errorMessage={errors.Status}
                  isInvalid={!!errors.Status}
                >
                  <SelectItem key="" disabled>
                    Select Status
                  </SelectItem>
                  <SelectItem key="Single">Single</SelectItem>
                  <SelectItem key="Married">Married</SelectItem>
                  <SelectItem key="Widowed">Widowed</SelectItem>
                  <SelectItem key="Separated">Separated</SelectItem>
                  <SelectItem key="Other">Other</SelectItem>
                </Select>
                <Input
                  type="text"
                  name="email"
                  label="Email"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.email}
                  onChange={(e) => setData({ ...data, email: e.target.value })}
                  errorMessage={errors.email}
                  isInvalid={!!errors.email}
                  endContent={
                    <FontAwesomeIcon
                      icon={faEnvelope}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Input
                  name="password"
                  label="Password"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.password}
                  onChange={(e) =>
                    setData({ ...data, password: e.target.value })
                  }
                  errorMessage={errors.password}
                  isInvalid={!!errors.password}
                  endContent={
                    <button
                      className="focus:outline-none"
                      type="button"
                      onClick={toggleVisibility}
                      aria-label="toggle password visibility"
                    >
                      {isVisible ? (
                        <FontAwesomeIcon
                          icon={faEyeSlash}
                          className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                        />
                      ) : (
                        <FontAwesomeIcon
                          icon={faEye}
                          className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                        />
                      )}
                    </button>
                  }
                  type={isVisible ? "text" : "password"}
                />
                <Input
                  name="confirmPassword"
                  label="Confirm Password"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  endContent={
                    <button
                      className="focus:outline-none"
                      type="button"
                      onClick={toggleVisibilityConfirm}
                      aria-label="toggle password visibility"
                    >
                      {isVisibleConfirm ? (
                        <FontAwesomeIcon
                          icon={faEyeSlash}
                          className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                        />
                      ) : (
                        <FontAwesomeIcon
                          icon={faEye}
                          className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                        />
                      )}
                    </button>
                  }
                  type={isVisibleConfirm ? "text" : "password"}
                  value={data.confirmPassword}
                  onChange={(e) =>
                    setData({ ...data, confirmPassword: e.target.value })
                  }
                  errorMessage={errors.confirmPassword}
                  isInvalid={!!errors.confirmPassword}
                />
              </div>
              <div
                className={`${
                  activeIndex === 1 ? "block" : "hidden"
                } grid grid-cols-1 md:grid-cols-2 gap-4`}
              >
                <Input
                  type="text"
                  name="StudentName"
                  label="Student First Name"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.StudentFirstName}
                  onChange={(e) =>
                    setData({ ...data, StudentFirstName: e.target.value })
                  }
                  errorMessage={errors.StudentFirstName}
                  isInvalid={!!errors.StudentFirstName}
                  endContent={
                    <FontAwesomeIcon
                      icon={faUser}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Input
                  type="text"
                  name="StudentName"
                  label="Student Last Name"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.StudentLastName}
                  onChange={(e) =>
                    setData({ ...data, StudentLastName: e.target.value })
                  }
                  errorMessage={errors.StudentLastName}
                  isInvalid={!!errors.StudentLastName}
                  endContent={
                    <FontAwesomeIcon
                      icon={faUser}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Input
                  type="text"
                  name="LRN"
                  label="LRN"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.LRN}
                  onChange={(e) => setData({ ...data, LRN: e.target.value })}
                  errorMessage={errors.LRN}
                  isInvalid={!!errors.LRN}
                  maxLength={12}
                  endContent={
                    <FontAwesomeIcon
                      icon={faIdCard}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
                <Input
                  className="pt-2"
                  type="date"
                  label="Birthday"
                  variant="bordered"
                  value={data.StudentBirthday}
                  onChange={(e) => {
                    const birthday = e.target.value;
                    const age = calculateAge(birthday); // Calculate age
                    setData({
                      ...data,
                      StudentBirthday: birthday,
                      StudentAge: age,
                    }); // Set both birthday and age
                  }}
                  errorMessage={errors.StudentBirthday}
                  isInvalid={!!errors.StudentBirthday}
                />
                <Select
                  label="Student Gender"
                  name="StudentGender"
                  placeholder="Select student gender"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.StudentGender || ""}
                  onChange={(e) =>
                    setData({ ...data, StudentGender: e.target.value })
                  }
                  errorMessage={errors.StudentGender}
                  isInvalid={!!errors.StudentGender}
                >
                  <SelectItem key="" disabled>
                    Select Gender
                  </SelectItem>
                  <SelectItem key="Male">Male</SelectItem>
                  <SelectItem key="Female">Female</SelectItem>
                  <SelectItem key="Other">Other</SelectItem>
                </Select>
                <Input
                  type="text"
                  name="MotherTongue"
                  label="Mother Tongue"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={data.MotherTongue}
                  onChange={(e) =>
                    setData({ ...data, MotherTongue: e.target.value })
                  }
                  errorMessage={errors.MotherTongue}
                  isInvalid={!!errors.MotherTongue}
                  endContent={
                    <FontAwesomeIcon
                      icon={faUser}
                      className="text-xl md:text-2xl text-default-400 pointer-events-none flex-shrink-0"
                    />
                  }
                />
              </div>
            </CardBody>

            <CardFooter className="flex flex-col md:flex-row justify-center gap-4 mt-6">
              <Button
                className="w-full md:w-1/3 text-md"
                color="danger"
                variant="light"
                onClick={() => navigate("/register")}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                className="w-full md:w-1/3 text-md"
                color="primary"
              >
                Register
              </Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </div>
  );
};

export default ParentRegister;
