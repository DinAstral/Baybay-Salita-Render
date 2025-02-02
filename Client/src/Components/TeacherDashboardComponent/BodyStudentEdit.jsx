/* eslint-disable react/no-unescaped-entities */
import { useState, useEffect } from "react";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import { useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

const BodyStudentEdit = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [data, setData] = useState({
    LRN: "",
    FirstName: "",
    MiddleName: "",
    LastName: "",
    Age: "",
    Level: "",
    Section: "",
    Birthday: "",
    Address: "",
    MotherTongue: "",
    Nationality: "",
    Gender: "",
    ContactNumber: "",
  });

  useEffect(() => {
    axios
      .get(`/api/getStudentID/${id}`)
      .then((response) => {
        setData(response.data);
      })
      .catch((err) => {
        console.error("Error fetching student data:", err);
        toast.error("Failed to fetch student data. Please try again later.");
      });
  }, [id]);

  const editStudent = async (e) => {
    e.preventDefault();
    try {
      const { data: responseData } = await axios.patch(
        `/api/updateStudent/${id}`,
        data
      );
      if (responseData.error) {
        toast.error(responseData.error);
      } else {
        toast.success("Updated Student info Successfully.");
        navigate("/manageStudent");
      }
    } catch (error) {
      console.error("Error updating student data:", error);
      toast.error("Failed to update student. Please try again.");
    }
  };

  return (
    <div className="container mx-auto px-6">
      <div className="max-w-9xl mx-auto bg-white shadow-lg rounded-lg p-8 pt">
        <div className="flex items-center gap-2 mb-6">
          <h2 className="text-2xl font-bold text-gray-800">
            Update Student's Information
          </h2>
          <Tooltip
            showArrow={true}
            content={
              <div className="px-2 py-1">
                <div className="text-sm font-bold">Update Information</div>
                <div className="text-xs">
                  This function will update the information of the students in
                  the system.
                </div>
              </div>
            }
          >
            <FontAwesomeIcon
              icon={faCircleInfo}
              size="md"
              className="text-gray-600"
            />
          </Tooltip>
        </div>
        <form onSubmit={editStudent}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">LRN</label>
              <Input
                underlined
                clearable
                value={data.LRN}
                onChange={(e) => setData({ ...data, LRN: e.target.value })}
                placeholder="Enter student's LRN"
                maxLength={12}
                type="number"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">First Name</label>
              <Input
                underlined
                clearable
                value={data.FirstName}
                onChange={(e) =>
                  setData({ ...data, FirstName: e.target.value })
                }
                placeholder="Enter the First Name"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Middle Name</label>
              <Input
                underlined
                clearable
                value={data.MiddleName}
                onChange={(e) =>
                  setData({ ...data, MiddleName: e.target.value })
                }
                placeholder="Enter the Middle Name"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Last Name</label>
              <Input
                underlined
                clearable
                value={data.LastName}
                onChange={(e) => setData({ ...data, LastName: e.target.value })}
                placeholder="Enter the Last Name"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Age</label>
              <Input
                underlined
                clearable
                value={data.Age}
                onChange={(e) => setData({ ...data, Age: e.target.value })}
                placeholder="Enter student's age"
                type="number"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Grade Level</label>
              <Select
                placeholder="Select Grade Level"
                value={data.Level}
                onChange={(value) => setData({ ...data, Level: value })}
              >
                <SelectItem key="" disabled>
                  Select Grade Level
                </SelectItem>
                <SelectItem key="Grade 1">Grade 1</SelectItem>
                {/* Add more options as needed */}
              </Select>
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Section</label>
              <Select
                placeholder="Select Section"
                value={data.Section}
                onChange={(value) => setData({ ...data, Section: value })}
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
              <label className="font-medium text-gray-700">Birthday</label>
              <Input
                underlined
                clearable
                value={data.Birthday}
                onChange={(e) => setData({ ...data, Birthday: e.target.value })}
                type="date"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Address</label>
              <Input
                underlined
                clearable
                value={data.Address}
                onChange={(e) => setData({ ...data, Address: e.target.value })}
                placeholder="Enter the Address"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Mother Tongue</label>
              <Input
                underlined
                clearable
                value={data.MotherTongue}
                onChange={(e) =>
                  setData({ ...data, MotherTongue: e.target.value })
                }
                placeholder="Enter the Mother Tongue"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Nationality</label>
              <Input
                underlined
                clearable
                value={data.Nationality}
                onChange={(e) =>
                  setData({ ...data, Nationality: e.target.value })
                }
                placeholder="Enter the Nationality"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Gender</label>
              <Select
                placeholder="Select Gender"
                value={data.Gender}
                onChange={(value) => setData({ ...data, Gender: value })}
              >
                <SelectItem key="" disabled>
                  Select Gender
                </SelectItem>
                <SelectItem key="male">Male</SelectItem>
                <SelectItem key="female">Female</SelectItem>
                <SelectItem key="other">Other</SelectItem>
              </Select>
            </div>
          </div>
          <div className="flex justify-end mt-6">
            <Button type="submit" color="primary" auto>
              Save Changes
            </Button>
            <Button
              auto
              flat
              color="danger"
              onClick={() => navigate(-1)}
              className="ml-4"
            >
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BodyStudentEdit;
