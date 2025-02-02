import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Tooltip, Button, Input, Select, SelectItem } from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import toast from "react-hot-toast";

const BodyAdminEditAdmin = () => {
  const navigate = useNavigate();
  const { id } = useParams();

  const [data, setData] = useState({
    FirstName: "",
    LastName: "",
    Age: "",
    Birthday: "",
    Gender: "",
    Address: "",
    Status: "",
    ContactNumber: "",
    email: "",
  });

  useEffect(() => {
    axios
      .get(`/api/getAdmin/${id}`) // route to include admin ID
      .then((response) => {
        setData(response.data);
      })
      .catch((err) => {
        console.error("Error fetching admin data:", err);
        toast.error("Failed to fetch admin data. Please try again later.");
      });
  }, [id]);

  const editAdmin = async (e) => {
    e.preventDefault();
    try {
      const { data: responseData } = await axios.patch(
        `/api/updateAdmin/${id}`,
        data
      );
      if (responseData.error) {
        toast.error(responseData.error);
      } else {
        toast.success("Updated admin information successfully.");
        navigate("/adminUsers");
      }
    } catch (error) {
      console.error("Error updating admin data:", error);
      toast.error("Failed to update admin. Please try again.");
    }
  };

  return (
    <div className="container mx-auto px-6">
      <div className="max-w-9xl mx-auto bg-white shadow-lg rounded-lg p-8">
        <div className="flex items-center gap-2 mb-6">
          <h2 className="text-2xl font-bold text-gray-800">
            Update Admin's Information
          </h2>
          <Tooltip
            showArrow={true}
            content={
              <div className="px-2 py-1">
                <div className="text-sm font-bold">Update Information</div>
                <div className="text-xs">
                  This function will update the information of the admin in the
                  system.
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
        <form onSubmit={editAdmin}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                placeholder="Enter Admin's age"
                type="number"
              />
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
              <label className="font-medium text-gray-700">Gender</label>
              <Select
                placeholder="Select Gender"
                value={data.Gender}
                onChange={(value) =>
                  setData({ ...data, Gender: e.target.value })
                }
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
              <label className="font-medium text-gray-700">Status</label>
              <Select
                placeholder="Select Status"
                value={data.Status}
                onChange={(value) =>
                  setData({ ...data, Status: e.target.value })
                }
              >
                <SelectItem key="" disabled>
                  Select Status
                </SelectItem>
                <SelectItem key="Single">Single</SelectItem>
                <SelectItem key="Married">Married</SelectItem>
                <SelectItem key="Widowed">Widowed</SelectItem>
                <SelectItem key="Separated">Separated</SelectItem>
              </Select>
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">
                Contact Number
              </label>
              <Input
                underlined
                clearable
                value={data.ContactNumber}
                onChange={(e) =>
                  setData({ ...data, ContactNumber: e.target.value })
                }
                placeholder="Enter the Contact Number"
              />
            </div>
            <div className="flex flex-col">
              <label className="font-medium text-gray-700">Email</label>
              <Input
                underlined
                clearable
                value={data.email}
                onChange={(e) => setData({ ...data, email: e.target.value })}
                placeholder="Enter the Admin's Email"
                type="email"
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

export default BodyAdminEditAdmin;
