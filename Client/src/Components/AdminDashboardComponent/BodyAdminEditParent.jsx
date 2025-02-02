/* eslint-disable react/no-unescaped-entities */
import { useState, useEffect } from "react";
import { Tooltip } from "@nextui-org/react";
import { useNavigate, useParams } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import "../ContentDasboard/Content.css";
import axios from "axios";
import toast from "react-hot-toast";

const BodyAdminEditParent = () => {
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
      .get(`/api/getUser/${id}`) // route to include student ID
      .then((response) => {
        console.log("Response:", response.data); // Log response data
        setData(response.data); // Assuming response.data contains student data
      })
      .catch((err) => {
        console.error("Error fetching student data:", err);
        toast.error("Failed to fetch student data. Please try again later.");
      });
  }, [id]); // Added 'id' to dependency array

  const editParent = async (e) => {
    e.preventDefault();
    const {
      FirstName,
      LastName,
      Age,
      Birthday,
      Address,
      Status,
      Gender,
      ContactNumber,
      email,
    } = data;
    try {
      const { data } = await axios.patch(`/api/updateParent/${id}`, {
        FirstName,
        LastName,
        Age,
        Birthday,
        Address,
        Status,
        Gender,
        ContactNumber,
        email,
      });
      if (data.error) {
        toast.error(data.error);
      } else {
        setData({});
        toast.success("Updated Parent information.");
        navigate("/adminUsers");
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="content-body">
      <div className="content-title-header">
        <div>
          Edit Parent's Information
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">Update Informaation</div>
                <div className="text-tiny">
                  This function will update the information of the parent in
                  system.
                </div>
              </div>
            }
          >
            <FontAwesomeIcon
              icon={faCircleInfo}
              size="1x"
              className="help-icon"
            />
          </Tooltip>
        </div>
      </div>
      <form onSubmit={editParent}>
        <div className="content-container">
          <div className="back-button-profile">
            <div className="btn-back" onClick={() => navigate(-1)}>
              Back
            </div>
          </div>
          <div className="add-inputs">
            <div className="add-input">
              <div className="label-add">First Name</div>
              <input
                type="text"
                name="addFirstName"
                id="addFirstName"
                placeholder="Enter the First Name"
                value={data.FirstName}
                onChange={(e) =>
                  setData({ ...data, FirstName: e.target.value })
                }
              />
            </div>
            <div className="add-input">
              <div className="label-add">Last Name</div>
              <input
                type="text"
                name="addLastName"
                id="addLastName"
                placeholder="Enter the Last Name"
                value={data.LastName}
                onChange={(e) => setData({ ...data, LastName: e.target.value })}
              />
            </div>
            <div className="add-input">
              <div className="label-add">Age</div>
              <input
                type="text"
                name="addAge"
                id="addAge"
                placeholder="Enter Teacher's age"
                value={data.Age}
                onChange={(e) => setData({ ...data, Age: e.target.value })}
              />
            </div>
            <div className="add-input">
              <div className="label-add">Birthday</div>
              <input
                type="date"
                name="addBirthday"
                id="addBirthday"
                placeholder="Enter Parent's Birthday"
                value={data.Birthday}
                onChange={(e) => setData({ ...data, Birthday: e.target.value })}
              />
            </div>
            <div className="add-input">
              <div className="label-add">Gender</div>
              <select
                className="select-gender"
                name="addGender"
                id="addGender"
                value={data.Gender}
                onChange={(e) => setData({ ...data, Gender: e.target.value })}
              >
                <option value="" disabled>
                  Select Gender
                </option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="add-input">
              <div className="label-add">Address</div>
              <input
                type="text"
                name="addAddress"
                id="addAddress"
                placeholder="Enter the Address"
                value={data.Address}
                onChange={(e) => setData({ ...data, Address: e.target.value })}
              />
            </div>
            <div className="add-input">
              <div className="label-add">Status</div>
              <input
                type="text"
                name="addStatus"
                id="addStatus"
                placeholder="Enter the Status"
                value={data.Status}
                onChange={(e) => setData({ ...data, Status: e.target.value })}
              />
            </div>
            <div className="add-input">
              <div className="label-add">Contact Number</div>
              <input
                type="text"
                name="addContactNumber"
                id="addContactNumber"
                placeholder="Enter the Contact Number"
                value={data.ContactNumber}
                onChange={(e) =>
                  setData({ ...data, ContactNumber: e.target.value })
                }
              />
            </div>
            <div className="add-input">
              <div className="label-add">Email</div>
              <input
                type="email"
                name="addEmail"
                id="addEmail"
                placeholder="Enter the Teacher's Email"
                value={data.email}
                onChange={(e) => setData({ ...data, email: e.target.value })}
              />
            </div>
          </div>
          <div className="add-student">
            <button className="btn-add" type="submit">
              Save Changes
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default BodyAdminEditParent;
