import { useEffect, useState, useRef, useContext } from "react";
import { useDownloadExcel } from "react-export-table-to-excel";
import ReactPaginate from "react-paginate";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo, faPrint } from "@fortawesome/free-solid-svg-icons";
import { UserContext } from "../../../context/userContext";
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
  Tooltip,
} from "@nextui-org/react";
import axios from "axios";
import "../ContentDasboard/Content.css";

import PrintRecord from "../Modals/PrintRecord";
import DeletePerformance from "../Modals/DeletePerformance";
import TeacherViewPerformance from "../Modals/TeacherViewPerformance";

const BodyManagePerformance = () => {
  const [modalShowPrint, setModalShowPrint] = useState(false);
  const [modalShowView, setModalShowView] = useState(false);
  const [modalShowDelete, setModalShowDelete] = useState(false);

  const [performances, setPerformances] = useState([]);
  const [selectedPerformance, setSelectedPerformance] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [usersPerPage] = useState(9);
  const [filteredPerformances, setFilteredPerformances] = useState([]);
  const [selectedSection, setSelectedSection] = useState("");
  const [selectedType, setSelectedType] = useState("");

  const tableRef = useRef(null);
  const { user } = useContext(UserContext);
  const [teacherSection, setTeacherSection] = useState(""); // Store teacher's section

  // Export all data except the "Actions" column
  const { onDownload } = useDownloadExcel({
    currentTableRef: tableRef.current,
    filename: "Performance_List_Report",
    sheet: "Performance",
    data: performances.map(({ _id, ...rest }) => rest), // Exclude "Actions" from the Excel export
  });
  // Fetch the teacher's section data
  useEffect(() => {
    if (user && user.UserID) {
      axios
        .get(`/api/getTeacher/${user.UserID}`) // API endpoint to fetch teacher data
        .then((response) => {
          setTeacherSection(response.data.Section); // Assuming the response contains the teacher's section as a string
        })
        .catch((err) => console.log(err));
    }
  }, [user]);

  // Fetch all performances and filter by teacher's section
  useEffect(() => {
    if (teacherSection) {
      axios
        .get("/api/getPerformance")
        .then((response) => {
          // Filter performances based on the logged-in teacher's section
          const filteredPerformances = response.data.filter(
            (performance) => performance.Section === teacherSection
          );
          setPerformances(filteredPerformances); // Store filtered performances
          setFilteredPerformances(filteredPerformances); // Set initial filtered data
        })
        .catch((err) => console.log(err));
    }
  }, [teacherSection]); // Trigger this useEffect when teacherSection is set

  // Handle page click for pagination
  const handlePageClick = (event) => {
    setCurrentPage(event.selected);
  };

  // Handle filtering by section or type
  const handleFilterChange = (e, filterType) => {
    const value = e.target.value;

    if (filterType === "section") {
      setSelectedSection(value);
    } else if (filterType === "type") {
      setSelectedType(value);
    }

    // Apply filtering based on selected section and type
    let filtered = performances;

    if (selectedSection) {
      filtered = filtered.filter(
        (performance) => performance.Section === selectedSection
      );
    }

    if (selectedType) {
      filtered = filtered.filter(
        (performance) => performance.Type === selectedType
      );
    }

    setFilteredPerformances(filtered); // Update filtered performances
  };

  const handlePrintClick = () => {
    setModalShowPrint(true);
  };

  const handleViewClick = (performance) => {
    setSelectedPerformance(performance);
    setModalShowView(true);
  };

  const handleDeleteClick = (performance) => {
    setSelectedPerformance(performance);
    setModalShowDelete(true);
  };

  // Get current set of performances for the current page
  const currentUsers = () => {
    const offset = currentPage * usersPerPage;
    return filteredPerformances.slice(offset, offset + usersPerPage);
  };

  // Calculate total pages for pagination
  const pageCount = () => {
    return Math.ceil(filteredPerformances.length / usersPerPage);
  };

  return (
    <div className="px-9">
      <PrintRecord
        show={modalShowPrint}
        onHide={() => setModalShowPrint(false)}
        print={onDownload}
      />
      <TeacherViewPerformance
        show={modalShowView}
        onHide={() => setModalShowView(false)}
        performance={selectedPerformance}
      />
      <DeletePerformance
        show={modalShowDelete}
        onHide={() => setModalShowDelete(false)}
        performance={selectedPerformance}
      />

      <div className="content-title-header">
        <div>
          Admin View Student Performance
          <Tooltip
            showArrow
            content={
              <div className="px-1 py-2">
                <div className="text-small font-bold">Performance Table</div>
                <div className="text-tiny">
                  This function will view the performance of your students in
                  each section.
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
            showArrow
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
              <div className="flex flex-row gap-2 mb-4">
                <Select
                  className="w-[20%]"
                  labelPlacement="outside"
                  label="Section"
                  variant="bordered"
                  defaultSelectedKeys={[""]}
                  onChange={(e) => handleFilterChange(e, "section")}
                  value={selectedSection}
                >
                  <SelectItem key="">Select Section</SelectItem>
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
                  className="w-[20%]"
                  labelPlacement="outside"
                  label="Type"
                  variant="bordered"
                  defaultSelectedKeys={[""]}
                  onChange={(e) => handleFilterChange(e, "type")}
                  value={selectedType}
                >
                  <SelectItem key="">Select Type of Assessment:</SelectItem>
                  <SelectItem key="Pagbabaybay">Assessment 1</SelectItem>
                  <SelectItem key="Pantig">Assessment 2</SelectItem>
                  <SelectItem key="Salita">Assessment 3</SelectItem>
                  <SelectItem key="Pagbabasa">Assessment 4</SelectItem>
                </Select>
              </div>

              <div className="card-body scrollable-table scrollable-container">
                <Table
                  ref={tableRef}
                  removeWrapper
                  color="primary"
                  selectionMode="single"
                >
                  <TableHeader>
                    <TableColumn>Input ID</TableColumn>
                    <TableColumn>LRN</TableColumn>
                    <TableColumn>Section</TableColumn>
                    <TableColumn>Activity Code</TableColumn>
                    <TableColumn>Type</TableColumn>
                    <TableColumn>Submitted</TableColumn>
                    <TableColumn className="text-center">Actions</TableColumn>
                  </TableHeader>
                  <TableBody emptyContent={"No rows to display."}>
                    {currentUsers().map((performance) => (
                      <TableRow key={performance._id}>
                        <TableCell>{performance.UserInputId}</TableCell>
                        <TableCell>{performance.LRN}</TableCell>
                        <TableCell>{performance.Section}</TableCell>
                        <TableCell>{performance.ActivityCode}</TableCell>
                        <TableCell>{performance.Type}</TableCell>
                        <TableCell>{performance.Result}</TableCell>
                        <TableCell className="text-center">
                          <div className="table-buttons">
                            <Button
                              color="primary"
                              onClick={() => handleViewClick(performance)}
                            >
                              View
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <ReactPaginate
                  previousLabel={"Previous"}
                  nextLabel={"Next"}
                  breakLabel={"..."}
                  breakClassName={"break-me"}
                  pageCount={pageCount()}
                  marginPagesDisplayed={2}
                  pageRangeDisplayed={5}
                  onPageChange={handlePageClick}
                  containerClassName={"pagination"}
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

export default BodyManagePerformance;
