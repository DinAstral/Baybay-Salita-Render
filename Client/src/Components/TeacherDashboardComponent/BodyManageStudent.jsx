import { useEffect, useState, useRef, useContext } from "react";
import { useDownloadExcel } from "react-export-table-to-excel";
import ReactPaginate from "react-paginate";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faSearch,
  faCircleInfo,
  faPrint,
} from "@fortawesome/free-solid-svg-icons";
import {
  Table,
  TableHeader,
  TableBody,
  TableColumn,
  TableRow,
  TableCell,
  Button,
  Select,
  SelectItem,
  Input,
  Tooltip,
} from "@nextui-org/react";
import axios from "axios";
import "../ContentDasboard/Content.css";
import PrintRecord from "../Modals/PrintRecord";
import ViewStudent from "../Modals/TeacherModal/ViewStudent";
import AddStudent from "../Modals/TeacherModal/AddStudent";
import UpdateStudent from "../Modals/TeacherModal/UpdateStudent";
import DeleteStudent from "../Modals/AdminModal/DeleteStudent";
import { UserContext } from "../../../context/userContext"; // Import the context

const BodyManageStudent = () => {
  const { user } = useContext(UserContext); // Get the logged-in user context
  const [modalShow, setModalShow] = useState(false);
  const [modalShowPrint, setModalShowPrint] = useState(false);
  const [modalShowEdit, setModalShowEdit] = useState(false);
  const [modalShowView, setModalShowView] = useState(false);
  const [modalShowDelete, setModalShowDelete] = useState(false);

  const [students, setStudents] = useState([]);
  const [filteredStudents, setFilteredStudents] = useState([]);
  const [teacherSection, setTeacherSection] = useState(""); // Store teacher's section
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [usersPerPage] = useState(10);
  const [selectedRole, setSelectedRole] = useState(""); // Filter state for Status
  const [searchQuery, setSearchQuery] = useState("");

  const tableRef = useRef(null);

  // Export all data except the "Actions" column
  const { onDownload } = useDownloadExcel({
    currentTableRef: tableRef.current,
    filename: "Student_List_Report",
    sheet: "Student",
    data: students.map(({ _id, ...rest }) => rest), // Exclude "Actions" from the Excel export
  });

  // Fetch the teacher's section data
  useEffect(() => {
    if (user && user.UserID) {
      axios
        .get(`/api/getTeacher/${user.UserID}`) // API endpoint to fetch teacher data
        .then((response) => {
          setTeacherSection(response.data.Section); // Assuming the response contains the teacher's section
        })
        .catch((err) => console.log(err));
    }
  }, [user]);

  // Fetch all students
  useEffect(() => {
    axios
      .get("/api/getStudents")
      .then((response) => {
        setStudents(response.data);
      })
      .catch((err) => console.log(err));
  }, []);

  // Filter students by section, search query, and status (selectedRole)
  useEffect(() => {
    let filtered = students;

    if (teacherSection) {
      // Filter students based on the teacher's section
      filtered = filtered.filter(
        (student) => student.Section === teacherSection
      );
    }

    if (searchQuery) {
      // Further filter by search query (FirstName, LastName, or LRN)
      filtered = filtered.filter(
        (student) =>
          student.FirstName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          student.LastName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          student.LRN.includes(searchQuery)
      );
    }

    if (selectedRole) {
      // Filter by status
      filtered = filtered.filter((student) => student.status === selectedRole);
    }

    setFilteredStudents(filtered);
  }, [searchQuery, selectedRole, students, teacherSection]);

  const handlePrintClick = () => setModalShowPrint(true);
  const handleViewClick = (student) => {
    setSelectedStudent(student);
    setModalShowView(true);
  };
  const handleEditClick = (student) => {
    setSelectedStudent(student);
    setModalShowEdit(true);
  };
  const handleDeleteClick = (student) => {
    setSelectedStudent(student);
    setModalShowDelete(true);
  };

  const handleDeleteSuccess = (deletedStudentId) => {
    setStudents(students.filter((student) => student._id !== deletedStudentId));
    setModalShowDelete(false);
  };

  const handlePageClick = (event) => setCurrentPage(event.selected);

  const offset = currentPage * usersPerPage;
  const currentUsers = filteredStudents.slice(offset, offset + usersPerPage);

  // Function to get color based on the status
  const getStatusColor = (status) => {
    switch (status) {
      case "Grade Ready Reader":
        return "text-green-600 font-semibold"; // Green with bold text
      case "Transitioning Reader":
        return "text-blue-600 font-semibold"; // Blue with bold text
      case "Developing Reader":
        return "text-orange-600 font-semibold"; // Orange with bold text
      case "Low Emerging Reader":
        return "text-red-600 font-semibold"; // Red with bold text
      case "High Emerging Reader":
        return "text-red-600 font-semibold"; // Red with bold text
      case "Incomplete":
        return "text-gray-600 italic"; // Gray with italic style
      default:
        return "text-black"; // Default color
    }
  };

  return (
    <div className="px-9">
      <PrintRecord
        show={modalShowPrint}
        onHide={() => setModalShowPrint(false)}
        print={onDownload}
      />
      <AddStudent show={modalShow} onHide={() => setModalShow(false)} />
      <ViewStudent
        show={modalShowView}
        onHide={() => setModalShowView(false)}
        student={selectedStudent}
      />
      <UpdateStudent
        show={modalShowEdit}
        student={selectedStudent}
        onHide={() => setModalShowEdit(false)}
      />
      <DeleteStudent
        show={modalShowDelete}
        onHide={() => setModalShowDelete(false)}
        student={selectedStudent}
        onDeleteSuccess={handleDeleteSuccess}
      />

      <div className="content-title-header">
        <div>
          Manage Student
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">Student Table</div>
                <div className="text-tiny">
                  This function will view the list of the students in each
                  section.
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
        <div className="generate-report">
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
              size="1x"
              className="print-icon"
              onClick={handlePrintClick}
            />
          </Tooltip>
        </div>
      </div>

      <div className="content-container">
        <div className="row">
          <div className="col">
            <div className="card mt-1 border-0">
              <div className="list-header-drop-score">
                <div className="flex flex-row gap-5 justify-start w-[100%]">
                  <div className="w-[40%] flex pt-4">
                    <Input
                      type="text"
                      className="pt-4"
                      placeholder="Search Student Name or LRN"
                      variant="bordered"
                      startContent={
                        <FontAwesomeIcon
                          icon={faSearch}
                          size="1x"
                          inverse
                          className="text-2xl text-default-400 pointer-events-none flex-shrink-0"
                        />
                      }
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>

                  <Select
                    className="w-[20%]"
                    labelPlacement="outside"
                    label="Status"
                    variant="bordered"
                    onChange={(e) => setSelectedRole(e.target.value)} // Add this line to update the selectedRole
                    defaultSelectedKeys={""}
                  >
                    <SelectItem key="">All</SelectItem>
                    <SelectItem key="Incomplete">Incomplete</SelectItem>
                    <SelectItem key="Low Emerging Reader">
                      Low Emerging
                    </SelectItem>
                    <SelectItem key="High Emerging Reader">
                      High Emerging
                    </SelectItem>
                    <SelectItem key="Developing Reader">Developing</SelectItem>
                    <SelectItem key="Transitioning Reader">
                      Transitioning
                    </SelectItem>
                    <SelectItem key="Grade Ready Reader">
                      Grade Ready
                    </SelectItem>
                  </Select>
                </div>
              </div>

              <div className="mt-4">
                <Table
                  ref={tableRef}
                  removeWrapper
                  color="primary"
                  selectionMode="single"
                  aria-label="Manage Students Table"
                >
                  <TableHeader>
                    <TableColumn>LRN</TableColumn>
                    <TableColumn>First Name</TableColumn>
                    <TableColumn>Last Name</TableColumn>
                    <TableColumn>Age</TableColumn>
                    <TableColumn>Section</TableColumn>
                    <TableColumn>Mother Tongue</TableColumn>
                    <TableColumn>Gender</TableColumn>
                    <TableColumn>Status</TableColumn>{" "}
                    {/* Added Status Column */}
                    <TableColumn className="text-center">Actions</TableColumn>
                  </TableHeader>
                  <TableBody emptyContent={"No rows to display."}>
                    {currentUsers.map((student) => (
                      <TableRow key={student._id}>
                        <TableCell>{student.LRN}</TableCell>
                        <TableCell>{student.FirstName}</TableCell>
                        <TableCell>{student.LastName}</TableCell>
                        <TableCell>{student.Age}</TableCell>
                        <TableCell>{student.Section}</TableCell>
                        <TableCell>{student.MotherTongue}</TableCell>
                        <TableCell>{student.Gender}</TableCell>
                        {/* Apply color to the status based on student performance */}
                        <TableCell>
                          <span className={getStatusColor(student.status)}>
                            {student.status}
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="flex justify-center gap-2">
                            <Button
                              color="default"
                              size="sm"
                              onClick={() => handleViewClick(student)}
                            >
                              View
                            </Button>
                            <Button
                              color="primary"
                              size="sm"
                              onClick={() => handleEditClick(student)}
                            >
                              Update
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                <ReactPaginate
                  previousLabel={"previous"}
                  nextLabel={"next"}
                  breakLabel={"..."}
                  breakClassName={"break-me"}
                  pageCount={Math.ceil(filteredStudents.length / usersPerPage)}
                  marginPagesDisplayed={2}
                  pageRangeDisplayed={5}
                  onPageChange={handlePageClick}
                  containerClassName={"pagination"}
                  subContainerClassName={"pages pagination"}
                  activeClassName={"active"}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BodyManageStudent;
