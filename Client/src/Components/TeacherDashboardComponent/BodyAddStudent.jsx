import React, { useState, useEffect, useContext } from "react";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import { useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import { UserContext } from "../../../context/userContext";
import axios from "axios";
import toast from "react-hot-toast";

const BodyTeacherAddStudent = () => {
  const navigate = useNavigate();
  const [data, setData] = useState({
    LRN: "",
    FirstName: "",
    MiddleName: "",
    LastName: "",
    Level: "",
    Section: "",
    Birthday: "",
    Address: "",
    MotherTongue: "",
    Gender: "",
    ContactNumber: "",
  });

  const [errors, setErrors] = useState({});
  const [verifiedParents, setVerifiedParents] = useState([]);
  const [filteredLRNs, setFilteredLRNs] = useState([]);
  const [teacherSection, setTeacherSection] = useState("");
  const [selectedStudent, setSelectedStudent] = useState(null);

  const { user } = useContext(UserContext);

  useEffect(() => {
    const fetchVerifiedParents = async () => {
      try {
        const response = await axios.get("/api/getParent");
        if (response.data && response.data.length > 0) {
          setVerifiedParents(response.data);
        } else {
          toast.error("No verified parents found.");
        }
      } catch (error) {
        console.error(error);
        toast.error("Failed to fetch parent data.");
      }
    };

    const fetchTeacherData = async () => {
      try {
        const response = await axios.get(`/api/getTeacher/${user.UserID}`);
        if (response.data && response.data.Section) {
          setTeacherSection(response.data.Section);
          setData((prevData) => ({
            ...prevData,
            Section: response.data.Section,
          }));
        } else {
          toast.error("Failed to fetch teacher's section.");
        }
      } catch (error) {
        console.error(error);
        toast.error("Failed to fetch teacher data.");
      }
    };

    fetchVerifiedParents();
    fetchTeacherData();
  }, []);

  // Handle search in LRN input
  const handleLRNSearch = (e) => {
    const inputLRN = e.target.value;
    setData({ ...data, LRN: inputLRN });

    // Filter verified students based on the LRN input
    if (inputLRN) {
      const results = verifiedParents
        .flatMap((parent) =>
          Array.isArray(parent.Student) ? parent.Student : []
        )
        .filter((student) => student.LRN.includes(inputLRN));
      setFilteredLRNs(results);
    } else {
      setFilteredLRNs([]);
    }
  };

  // Handle LRN selection and auto-populate student data
  const handleLRNSelect = (selectedLRN) => {
    setData({ ...data, LRN: selectedLRN });
    setFilteredLRNs([]);

    const student = verifiedParents
      .flatMap((parent) =>
        Array.isArray(parent.Student) ? parent.Student : []
      )
      .find((student) => student.LRN === selectedLRN);

    if (student) {
      const parent = verifiedParents.find(
        (p) =>
          Array.isArray(p.Student) &&
          p.Student.some((s) => s.LRN === selectedLRN)
      );

      setSelectedStudent(student);
      setData({
        LRN: student.LRN,
        FirstName: student.FirstName,
        LastName: student.LastName,
        Birthday: student.Birthday,
        Age: calculateAge(student.Birthday),
        Address: parent?.Address || "",
        MotherTongue: student.MotherTongue,
        Gender: student.Gender,
        ContactNumber: parent?.ContactNumber || "",
        Section: teacherSection,
      });
    } else {
      toast.error("Student not found.");
    }
  };

  // Handle Birthday input change and automatically update Age
  const handleBirthdayChange = (e) => {
    const birthday = e.target.value;
    setData({
      ...data,
      Birthday: birthday,
      Age: calculateAge(birthday), // Automatically calculate Age on birthday change
    });
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

  // Handle other input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setData({ ...data, [name]: value });
  };

  // Validate inputs before adding the student
  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    if (!data.LRN) {
      newErrors.LRN = "LRN is required.";
      isValid = false;
    }

    if (!data.FirstName) {
      newErrors.FirstName = "First Name is required.";
      isValid = false;
    }

    if (!data.LastName) {
      newErrors.LastName = "Last Name is required.";
      isValid = false;
    }

    if (!data.Birthday) {
      newErrors.Birthday = "Birthday is required.";
      isValid = false;
    }

    if (!data.Address) {
      newErrors.Address = "Address is required.";
      isValid = false;
    }

    if (!data.MotherTongue) {
      newErrors.MotherTongue = "Mother Tongue is required.";
      isValid = false;
    }

    if (!data.Gender) {
      newErrors.Gender = "Gender is required.";
      isValid = false;
    }

    if (!data.ContactNumber) {
      newErrors.ContactNumber = "Contact Number is required.";
      isValid = false;
    }

    if (!data.Section) {
      newErrors.Section = "Section is required.";
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  // Handle form submission
  const addStudent = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out the form correctly.");
      return;
    }

    try {
      const response = await axios.post("/api/addStudent", data);
      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({
          LRN: "",
          FirstName: "",
          MiddleName: "",
          LastName: "",
          Level: "",
          Section: teacherSection,
          Birthday: "",
          Age: "",
          Address: "",
          MotherTongue: "",
          Gender: "",
          ContactNumber: "",
        });
        toast.success("Student added successfully.");
        navigate("/manageStudent");
      }
    } catch (error) {
      console.error(error);
      toast.error("Failed to add student. Please try again.");
    }
  };

  return (
    <div className="container mx-auto px-6 pt-[4rem]">
      <div className="bg-white shadow-md rounded-md px-8 pt-6 pb-8 mb-4 flex flex-col">
        <h2 className="mb-6 text-3xl font-bold text-gray-700 flex items-center gap-2">
          Add Student's Information
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">Add Student</div>
                <div className="text-tiny">
                  This function will add the information of the students in the
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
        <form onSubmit={addStudent}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* LRN Input with Search and Suggestions */}
            <div className="relative">
              <Input
                label="LRN"
                placeholder="Enter or search LRN"
                value={data.LRN}
                onChange={handleLRNSearch}
                errorMessage={errors.LRN}
                isInvalid={!!errors.LRN}
              />
              {filteredLRNs.length > 0 && (
                <div className="absolute z-10 bg-white border border-gray-200 rounded-md shadow-lg mt-1 w-full max-h-48 overflow-y-auto">
                  {filteredLRNs.map((student) => (
                    <div
                      key={student.LRN}
                      onClick={() => handleLRNSelect(student.LRN)}
                      className="px-4 py-2 hover:bg-gray-100 cursor-pointer"
                    >
                      <span className="font-semibold">{student.LRN}</span> -{" "}
                      {student.FirstName} {student.LastName}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <Input
              label="First Name"
              type="text"
              placeholder="Enter the First Name"
              name="FirstName"
              value={data.FirstName}
              onChange={handleChange}
              errorMessage={errors.FirstName}
              isInvalid={!!errors.FirstName}
            />
            <Input
              label="Last Name"
              type="text"
              placeholder="Enter the Last Name"
              name="LastName"
              value={data.LastName}
              onChange={handleChange}
              errorMessage={errors.LastName}
              isInvalid={!!errors.LastName}
            />
            <Input
              label="Birthday"
              type="date"
              name="Birthday"
              value={data.Birthday.split("T")[0]} // Format date
              onChange={handleChange}
              errorMessage={errors.Birthday}
              isInvalid={!!errors.Birthday}
            />
            <Input
              label="Birthday"
              type="date"
              name="Birthday"
              value={data.Birthday.split("T")[0]} // Format date
              onChange={handleBirthdayChange} // Handle birthday change
              errorMessage={errors.Birthday}
              isInvalid={!!errors.Birthday}
            />
            <Input
              label="Age"
              type="text"
              name="Age"
              value={data.Age}
              isDisabled={true} // Age field is read-only
            />
            <Input
              label="Mother Tongue"
              type="text"
              placeholder="Enter the Mother Tongue"
              name="MotherTongue"
              value={data.MotherTongue}
              onChange={handleChange}
              errorMessage={errors.MotherTongue}
              isInvalid={!!errors.MotherTongue}
            />
            {/* Gender */}
            <Input
              label="Gender"
              type="text"
              placeholder="Enter Gender"
              name="Gender"
              value={data.Gender}
              onChange={handleChange}
              errorMessage={errors.Gender}
              isInvalid={!!errors.Gender}
            />
            <Input
              label="Contact Number"
              type="text"
              placeholder="Enter the Contact Number"
              name="ContactNumber"
              value={data.ContactNumber}
              onChange={handleChange}
              errorMessage={errors.ContactNumber}
              isInvalid={!!errors.ContactNumber}
            />
            <Input
              label="Section"
              type="text"
              placeholder="Enter the Section"
              name="Section"
              value={data.Section}
              onChange={handleChange}
              errorMessage={errors.Section}
              isInvalid={!!errors.Section}
              readOnly // Make Section read-only
            />
          </div>
          <div className="mt-6">
            <Button type="submit" color="primary" size="lg" radius="sm">
              Add Student
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BodyTeacherAddStudent;
