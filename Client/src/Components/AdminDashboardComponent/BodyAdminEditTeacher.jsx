/* eslint-disable react/no-unescaped-entities */
import { useState, useEffect } from "react";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import { useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

const BodyAdminEditTeacher = () => {
  const navigate = useNavigate();
  const { id } = useParams();

  const [data, setData] = useState({
    UserID: "",
    FirstName: "",
    MiddleName: "",
    LastName: "",
    Department: "",
    Age: "",
    Section: "",
    Birthday: "",
    Address: "",
    Nationality: "",
    Gender: "",
    ContactNumber: "",
    email: "",
  });

  useEffect(() => {
    axios
      .get(`/api/getUser/${id}`)
      .then((response) => {
        setData(response.data);
      })
      .catch((err) => {
        console.error("Error fetching teacher data:", err);
        toast.error("Failed to fetch teacher data. Please try again later.");
      });
  }, [id]);

  const editTeacher = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.patch(`/api/updateTeacher/${id}`, data);
      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        toast.success("Updated Teacher info successfully.");
        navigate("/adminUsers");
      }
    } catch (error) {
      console.error("Error updating teacher data:", error);
      toast.error("Failed to update teacher. Please try again.");
    }
  };

  return (
    <div className="max-w-4xl mx-auto bg-white shadow-lg rounded-lg p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">
          Update Teacher's Information
        </h2>
        <Tooltip
          showArrow={true}
          content={
            <div className="px-1 py-2">
              <div className="text-sm font-bold">Update Information</div>
              <div className="text-xs">
                This function will update the information of the teacher in the
                system.
              </div>
            </div>
          }
        >
          <FontAwesomeIcon
            icon={faCircleInfo}
            size="lg"
            className="text-gray-600"
          />
        </Tooltip>
      </div>
      <form onSubmit={editTeacher}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">ID Number</label>
            <Input
              underlined
              disabled
              value={data.UserID}
              onChange={(e) => setData({ ...data, UserID: e.target.value })}
            />
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">First Name</label>
            <Input
              underlined
              value={data.FirstName}
              onChange={(e) => setData({ ...data, FirstName: e.target.value })}
              placeholder="Enter the First Name"
            />
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Middle Name</label>
            <Input
              underlined
              value={data.MiddleName}
              onChange={(e) => setData({ ...data, MiddleName: e.target.value })}
              placeholder="Enter the Middle Name"
            />
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Last Name</label>
            <Input
              underlined
              value={data.LastName}
              onChange={(e) => setData({ ...data, LastName: e.target.value })}
              placeholder="Enter the Last Name"
            />
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Age</label>
            <Input
              underlined
              value={data.Age}
              onChange={(e) => setData({ ...data, Age: e.target.value })}
              placeholder="Enter Teacher's age"
              type="number"
            />
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Department</label>
            <Select
              value={data.Department}
              onChange={(value) =>
                setData({ ...data, Department: e.target.value })
              }
            >
              <SelectItem key="" disabled>
                Select Department
              </SelectItem>
              <SelectItem key="Filipino Department">
                Filipino Department
              </SelectItem>
              {/* Add more options as needed */}
            </Select>
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Section</label>
            <Select
              value={data.Section}
              onChange={(value) =>
                setData({ ...data, Section: e.target.value })
              }
            >
              <SelectItem key="" disabled>
                Select Section
              </SelectItem>
              <SelectItem key="Camia">Camia</SelectItem>
              <SelectItem key="Daffodil">Daffodil</SelectItem>
              <SelectItem key="Daisy">Daisy</SelectItem>
              <SelectItem key="Gumamela">Gumamela</SelectItem>
              <SelectItem key="Lily">Lily</SelectItem>
              <SelectItem key="Rosal">Rosal</SelectItem>
              <SelectItem key="Rose">Rose</SelectItem>
              <SelectItem key="Santan">Santan</SelectItem>
              <SelectItem key="Special">Special</SelectItem>
            </Select>
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Email</label>
            <Input
              underlined
              value={data.email}
              onChange={(e) => setData({ ...data, email: e.target.value })}
              placeholder="Enter the Teacher's Email"
              type="email"
            />
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Gender</label>
            <Select
              value={data.Gender}
              onChange={(value) => setData({ ...data, Gender: e.target.value })}
            >
              <SelectItem key="" disabled>
                Select Gender
              </SelectItem>
              <SelectItem key="Male">Male</SelectItem>
              <SelectItem key="Female">Female</SelectItem>
              <SelectItem key="Other">Other</SelectItem>
            </Select>
          </div>
          <div className="flex flex-col">
            <label className="font-medium text-gray-700">Contact Number</label>
            <Input
              underlined
              value={data.ContactNumber}
              onChange={(e) =>
                setData({ ...data, ContactNumber: e.target.value })
              }
              placeholder="Enter the Contact Number"
            />
          </div>
        </div>
        <div className="flex justify-end mt-6">
          <Button type="submit" color="primary" auto>
            Save Changes
          </Button>
          <Button
            auto
            flat
            color="error"
            onClick={() => navigate(-1)}
            className="ml-4"
          >
            Cancel
          </Button>
        </div>
      </form>
    </div>
  );
};

export default BodyAdminEditTeacher;
