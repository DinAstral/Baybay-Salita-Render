import { useEffect, useState, useRef } from "react";
import "../ContentDasboard/Content.css"; // Ensure CSS handles responsive layouts
import PrintRecord from "../Modals/PrintRecord";
import AddUser from "../Modals/AdminModal/AddUserModal";
import EditUser from "../Modals/AdminModal/EditUser";
import ViewParent from "../Modals/AdminModal/ViewParent";
import ViewTeacher from "../Modals/AdminModal/ViewTeacher";
import UpdateParent from "../Modals/AdminModal/UpdateParent";
import UpdateTeacher from "../Modals/AdminModal/UpdateTeacher";
import DeleteUser from "../Modals/AdminModal/DeleteUser";
import { useDownloadExcel } from "react-export-table-to-excel";
import { Tooltip } from "@nextui-org/react";
import ReactPaginate from "react-paginate";
import axios from "axios";
import {
  Table,
  TableHeader,
  TableBody,
  TableColumn,
  TableRow,
  TableCell,
  Button,
  Input,
  Select,
  SelectItem,
} from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faSearch,
  faCircleInfo,
  faPrint,
} from "@fortawesome/free-solid-svg-icons";

// Utility functions for filtering and paginating users
const getFilteredUsers = (users, selectedRole, searchQuery) => {
  let filtered = users;
  if (selectedRole !== "default") {
    filtered = filtered.filter((user) => user.role === selectedRole);
  }
  if (searchQuery) {
    filtered = filtered.filter((user) =>
      user.email.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }
  return filtered;
};

const getCurrentUsers = (filteredUsers, currentPage, usersPerPage) => {
  const offset = currentPage * usersPerPage;
  return filteredUsers.slice(offset, offset + usersPerPage);
};

const BodyAdminUsers = () => {
  const [modalState, setModalState] = useState({
    add: false,
    print: false,
    editUser: false,
    editTeacher: false,
    viewTeacher: false,
    editParent: false,
    viewParent: false,
    delete: false,
  });
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [usersPerPage] = useState(10);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRole, setSelectedRole] = useState("default");

  const tableRef = useRef(null);
  // Export all data except the "Actions" column
  const { onDownload } = useDownloadExcel({
    currentTableRef: tableRef.current,
    filename: "User_List_Report",
    sheet: "User",
    data: users.map(({ _id, ...rest }) => rest), // Exclude "Actions" from the Excel export
  });

  // Fetch teacher and parent data
  const fetchUsers = async () => {
    try {
      const [teachers, parents] = await Promise.all([
        axios.get("/api/getTeacher"),
        axios.get("/api/getParent"),
      ]);
      const teacherData = teachers.data.map((teacher) => ({
        ...teacher,
        role: "Teacher",
      }));
      const parentData = parents.data.map((parent) => ({
        ...parent,
        role: "Parent",
      }));
      setUsers([...teacherData, ...parentData]);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  // Apply filters and pagination
  const filteredUsers = getFilteredUsers(users, selectedRole, searchQuery);
  const currentUsers = getCurrentUsers(
    filteredUsers,
    currentPage,
    usersPerPage
  );

  const handleModalStateChange = (modalType, value) => {
    setModalState((prev) => ({ ...prev, [modalType]: value }));
  };

  const handlePageClick = (event) => {
    setCurrentPage(event.selected);
  };

  return (
    <div className="px-4 md:px-9">
      <AddUser
        show={modalState.add}
        onHide={() => handleModalStateChange("add", false)}
      />
      <EditUser
        show={modalState.editUser}
        onHide={() => handleModalStateChange("editUser", false)}
        user={selectedUser}
      />
      <PrintRecord
        show={modalState.print}
        onHide={() => handleModalStateChange("print", false)}
        print={onDownload}
      />
      <ViewTeacher
        show={modalState.viewTeacher}
        user={selectedUser}
        onHide={() => handleModalStateChange("viewTeacher", false)}
      />
      <UpdateTeacher
        show={modalState.editTeacher}
        user={selectedUser}
        onHide={() => handleModalStateChange("editTeacher", false)}
      />
      <ViewParent
        show={modalState.viewParent}
        user={selectedUser}
        onHide={() => handleModalStateChange("viewParent", false)}
      />
      <UpdateParent
        show={modalState.editParent}
        user={selectedUser}
        onHide={() => handleModalStateChange("editParent", false)}
      />
      <DeleteUser
        show={modalState.delete}
        onHide={() => handleModalStateChange("delete", false)}
        user={selectedUser}
        onDeleteSuccess={fetchUsers}
      />

      <div className="flex justify-between items-center mb-6 px-2">
        <div className="text-2xl md:text-3xl font-semibold">
          Manage User
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">User Table</div>
                <div className="text-tiny">
                  This function will view the information of the students in the
                  system.
                </div>
              </div>
            }
          >
            <FontAwesomeIcon
              icon={faCircleInfo}
              size="1x"
              className="ml-2 cursor-pointer text-lg"
            />
          </Tooltip>
        </div>
        <div className="cursor-pointer flex items-center">
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">Print</div>
                <div className="text-tiny">Generate report/Print table</div>
              </div>
            }
          >
            <FontAwesomeIcon
              icon={faPrint}
              className=" pr-10 cursor-pointer text-lg size-8"
              onClick={() => handleModalStateChange("print", true)}
            />
          </Tooltip>
        </div>
      </div>

      <div className="max-w-full min-h-[75vh] bg-white p-4 md:p-6 rounded-3xl">
        <div className="flex flex-col md:flex-row justify-between items-center mb-4">
          <Select
            labelPlacement="outside"
            label="Role"
            variant="bordered"
            className="bg-transparent w-full md:w-1/5"
            onChange={(e) => setSelectedRole(e.target.value)}
            value={selectedRole}
          >
            <SelectItem key="default">All</SelectItem>
            <SelectItem key="Admin">Admin</SelectItem>
            <SelectItem key="Parent">Parent</SelectItem>
            <SelectItem key="Teacher">Teacher</SelectItem>
          </Select>

          <div className="w-full md:w-[40%] flex items-center mt-4 md:mt-4">
            <Input
              variant="bordered"
              type="text"
              placeholder="Enter User Email"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              startContent={
                <FontAwesomeIcon
                  icon={faSearch}
                  size="1x"
                  inverse
                  className="text-2xl text-default-400 pointer-events-none flex-shrink-0"
                />
              }
            />
          </div>

          <div className="mt-4 md:mt-4">
            <Button
              color="primary"
              className="w-full md:w-[120px] text-md"
              onClick={() => handleModalStateChange("add", true)}
            >
              Add User
            </Button>
          </div>
        </div>

        <div className="overflow-auto max-h-[600px]">
          <Table
            aria-label="User Table"
            ref={tableRef}
            removeWrapper
            color="primary"
            selectionMode="single"
            className="w-full whitespace-nowrap"
          >
            <TableHeader>
              <TableColumn>UserID</TableColumn>
              <TableColumn>Email</TableColumn>
              <TableColumn>Role</TableColumn>
              <TableColumn className="text-center">Actions</TableColumn>
            </TableHeader>
            <TableBody emptyContent={"No rows to display."}>
              {currentUsers.map((user) => (
                <TableRow key={user.UserID}>
                  <TableCell>{user.UserID}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>{user.role}</TableCell>
                  <TableCell className="text-center">
                    <div className="flex justify-center space-x-2">
                      <Button
                        color="default"
                        onClick={() => {
                          setSelectedUser(user);
                          if (user.role === "Teacher") {
                            handleModalStateChange("viewTeacher", true);
                          } else if (user.role === "Parent") {
                            handleModalStateChange("viewParent", true);
                          }
                        }}
                      >
                        View
                      </Button>
                      <Button
                        color="primary"
                        onClick={() =>
                          setSelectedUser(user) ||
                          handleModalStateChange("editUser", true)
                        }
                      >
                        Update
                      </Button>
                      <Button
                        className="bg-[#ff505b] text-white"
                        onClick={() => {
                          setSelectedUser(user);
                          handleModalStateChange("delete", true);
                        }}
                      >
                        Delete
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <ReactPaginate
          pageCount={Math.ceil(filteredUsers.length / usersPerPage)}
          onPageChange={handlePageClick}
          containerClassName={"flex justify-center mt-4 space-x-2"}
          activeClassName={
            "text-white bg-blue-600 hover:bg-gray-200 px-3 py-1 rounded"
          }
          pageLinkClassName={"px-2 py-1 rounded"}
          previousLinkClassName={"px-2 py-1  rounded"}
          nextLinkClassName={"px-2 py-1 rounded"}
        />
      </div>
    </div>
  );
};

export default BodyAdminUsers;
