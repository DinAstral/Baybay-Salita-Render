import { useState, useEffect, useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faChalkboardTeacher,
  faChildReaching,
  faCircleInfo,
  faHouse,
} from "@fortawesome/free-solid-svg-icons";
import { Tooltip, Select, SelectItem } from "@nextui-org/react";
import { BarChart } from "@mui/x-charts/BarChart";
import axios from "axios";

const BodyAdminDashboard = () => {
  const [students, setStudents] = useState([]);
  const [assessments, setAssessment] = useState([]);
  const [teachersCount, setTeachersCount] = useState(0);
  const [parentsCount, setParentsCount] = useState(0);
  const [selectedAssessment, setSelectedAssessment] = useState("Pagbabaybay");
  const [performanceCounts, setPerformanceCounts] = useState([]);

  const [averageScores, setAverageScores] = useState({});
  const [selectedSection, setSelectedSection] = useState("All Sections");

  const [sectionStatusCounts, setSectionStatusCounts] = useState({});
  const [sectionRecommendations, setSectionRecommendations] = useState({});

  // Function to handle section selection
  const handleSectionClick = (section) => {
    setSelectedSection(section);
  };

  const sections = [
    "Aster",
    "Camia",
    "Dahlia",
    "Iris",
    "Jasmin",
    "Orchid",
    "Rose",
    "Tulip",
    "SSC",
  ];

  const [sectionCounts, setSectionCounts] = useState({
    Aster: 0,
    Camia: 0,
    Dahlia: 0,
    Iris: 0,
    Jasmin: 0,
    Orchid: 0,
    Rose: 0,
    Tulip: 0,
    SSC: 0,
  });

  const [activityCounts, setActivityCounts] = useState({
    Aster: 0,
    Camia: 0,
    Dahlia: 0,
    Iris: 0,
  });

  const [statusCounts, setStatusCounts] = useState({
    Incomplete: 0,
    LowEmergingReader: 0,
    HighEmergingReader: 0,
    DevelopingReader: 0,
    TransitioningReader: 0,
    GradeLevelReader: 0,
  });

  // Fetch data when the component loads or when the selectedAssessment changes
  useEffect(() => {
    const fetchData = async () => {
      try {
        await Promise.all([fetchUsers(), fetchStudents(), fetchAssessment()]);
        await fetchPerformance(); // Fetch performance data separately
        computeAverageScores();
        generateSectionRecommendations();
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    };

    fetchData();
  }, [selectedAssessment]);

  const fetchUsers = async () => {
    try {
      const response = await axios.get("/api/users");
      const data = response.data;
      setTeachersCount(countByRole(data, "Teacher"));
      setParentsCount(countByRole(data, "Parent"));
    } catch (err) {
      console.error("Error fetching users:", err);
    }
  };

  const fetchStudents = async () => {
    try {
      const response = await axios.get("/api/getStudents");
      const data = response.data;

      setStudents(data);

      // Group students by their status and update the state
      const statusGroups = {
        Incomplete: 0,
        LowEmergingReader: 0,
        HighEmergingReader: 0,
        DevelopingReader: 0,
        TransitioningReader: 0,
        GradeLevelReader: 0,
      };

      data.forEach((student) => {
        switch (student.status) {
          case "Incomplete":
            statusGroups.Incomplete += 1;
            break;
          case "Low Emerging Reader":
            statusGroups.LowEmergingReader += 1;
            break;
          case "High Emerging Reader":
            statusGroups.HighEmergingReader += 1;
            break;
          case "Developing Reader":
            statusGroups.DevelopingReader += 1;
            break;
          case "Transitioning Reader":
            statusGroups.TransitioningReader += 1;
            break;
          case "Grade Level Reader":
            statusGroups.GradeLevelReader += 1;
            break;
          default:
            break;
        }
      });

      setStatusCounts(statusGroups);

      const sectionCounts = countStudentsBySection(data);
      setSectionCounts(sectionCounts.totalCounts); // Set total counts
      setSectionStatusCounts(sectionCounts.statusCounts); // Set status counts
    } catch (err) {
      console.error("Error fetching students:", err);
    }
  };

  const fetchAssessment = async () => {
    try {
      const response = await axios.get("/api/getAssessments");
      const data = response.data;

      const activities = ["Pagbabaybay", "Pantig", "Salita", "Pagbabasa"];
      const counts = activities.reduce((acc, activity) => {
        acc[activity] = data.filter((type) => type.Type === activity).length;
        return acc;
      }, {});

      setAssessment(data);
      setActivityCounts(counts);
    } catch (err) {
      console.error("Error fetching assessments:", err);
    }
  };

  const fetchPerformance = async () => {
    try {
      const response = await axios.get("/api/getPerformance");
      const data = response.data;

      setPerformanceCounts(data);
    } catch (err) {
      console.error("Error fetching performance:", err);
    }
  };

  useEffect(() => {
    if (students.length && performanceCounts.length) {
      computeAverageScores();
      generateSectionRecommendations();
    }
  }, [students, performanceCounts, selectedAssessment]);

  // Count the number of users by role (e.g., Teacher, Parent)
  const countByRole = (data, role) =>
    data.filter((user) => user.role === role).length;

  // Function to count students by section and status
  const countStudentsBySection = (students) => {
    const sections = [
      "Aster",
      "Camia",
      "Dahlia",
      "Iris",
      "Jasmin",
      "Orchid",
      "Rose",
      "Tulip",
      "SSC",
    ];

    const statusTypes = [
      "Incomplete",
      "Low Emerging Reader",
      "High Emerging Reader",
      "Developing Reader",
      "Transitioning Reader",
      "Grade Level Reader",
    ];

    const totalCounts = {};
    const statusCounts = {};

    sections.forEach((section) => {
      totalCounts[section] = 0;
      statusCounts[section] = {};

      statusTypes.forEach((status) => {
        const count = students.filter(
          (student) => student.Section === section && student.status === status
        ).length;

        statusCounts[section][status] = count;
        totalCounts[section] += count;
      });
    });

    return { totalCounts, statusCounts };
  };

  // Assign colors to statuses
  const getColorForStatus = (status) => {
    switch (status) {
      case "Incomplete":
        return "#ff505b";
      case "Low Emerging Reader":
        return "#ff7828";
      case "High Emerging Reader":
        return "#DC84F3";
      case "Developing Reader":
        return "#4b99f5";
      case "Transitioning Reader":
        return "#ffce1f";
      case "Grade Level Reader":
        return "#FF8080";
      default:
        return "#ccc"; // Default color for unknown status
    }
  };

  // Data preparation for the bar charts
  const sectionLabels = Object.keys(sectionCounts);
  const statusTypes = [
    "Incomplete",
    "Low Emerging Reader",
    "High Emerging Reader",
    "Developing Reader",
    "Transitioning Reader",
    "Grade Level Reader",
  ];

  // Create data series for the BarChart
  const statusSeries = statusTypes.map((status) => ({
    data: sectionLabels.map(
      (section) =>
        (sectionStatusCounts[section] &&
          sectionStatusCounts[section][status]) ||
        0
    ),
    label: status,
    color: getColorForStatus(status),
  }));

  const computeAverageScores = () => {
    const sectionScores = {};

    // Initialize each section with an array to accumulate scores
    sections.forEach((section) => {
      sectionScores[section] = [];
    });

    // Filter performance counts based on selected assessment
    performanceCounts.forEach((performance) => {
      const student = students.find((s) => s.LRN === performance.LRN);
      if (student && student.Section) {
        const section = student.Section;
        if (performance.Type === selectedAssessment) {
          sectionScores[section].push(parseInt(performance.Score, 10));
        }
      }
    });

    // Calculate the average score per section
    const avgScores = {};
    sections.forEach((section) => {
      const scores = sectionScores[section];
      avgScores[section] = scores.length
        ? scores.reduce((acc, score) => acc + score, 0) / scores.length
        : 0;
    });

    setAverageScores(avgScores);
  };

  const generateSectionRecommendations = () => {
    const recommendations = {};

    sections.forEach((section) => {
      const sectionPerformances = performanceCounts.filter((performance) => {
        const student = students.find((s) => s.LRN === performance.LRN);
        return student && student.Section === section;
      });

      const wordErrorCounts = {}; // Track incorrect words
      const letterErrorCounts = {}; // Track incorrect letters
      const statusReasons = {
        Incomplete: [],
        LowEmergingReader: [],
        HighEmergingReader: [],
        DevelopingReader: [],
        TransitioningReader: [],
        GradeLevelReader: [],
      };

      sectionPerformances.forEach((performance) => {
        // Track status reasons
        const status = performance.Status;
        if (status === "Incomplete") {
          statusReasons[status].push(
            `The student has not yet finished answering the ${performance.Type} assessment.`
          );
        } else if (statusReasons[status]) {
          statusReasons[status].push(
            `Low scores in ${performance.Type} (Score: ${performance.Score})`
          );
        }

        // Track word and letter errors
        performance.PerformanceItems.forEach((item) => {
          if (item.Remarks.toLowerCase() === "incorrect") {
            // Track word errors
            wordErrorCounts[item.Word] = (wordErrorCounts[item.Word] || 0) + 1;

            // Track letter errors (if applicable)
            if (item.Word.length === 1) {
              letterErrorCounts[item.Word] =
                (letterErrorCounts[item.Word] || 0) + 1;
            }
          }
        });
      });

      // Get top 5 most challenging words
      const topWords = Object.keys(wordErrorCounts)
        .sort((a, b) => wordErrorCounts[b] - wordErrorCounts[a])
        .slice(0, 5);

      // Get top 5 most challenging letters (if applicable)
      const topLetters = Object.keys(letterErrorCounts)
        .sort((a, b) => letterErrorCounts[b] - letterErrorCounts[a])
        .slice(0, 5);

      // Generate recommendations
      const sectionRecommendations = [];

      // Add status reasons
      Object.keys(statusReasons).forEach((status) => {
        if (statusReasons[status].length > 0) {
          if (status === "Incomplete") {
            sectionRecommendations.push(
              `Students with status "${status}" have not yet finished their assessments.`
            );
          } else {
            sectionRecommendations.push(
              `Students with status "${status}" have the following issues: ${statusReasons[
                status
              ].join(", ")}.`
            );
          }
        }
      });

      // Add word and letter difficulties (only for non-Incomplete statuses)
      if (selectedSection !== "Incomplete") {
        if (topWords.length > 0) {
          sectionRecommendations.push(
            `Students are struggling with the following words: ${topWords.join(
              ", "
            )}.`
          );
        }

        if (topLetters.length > 0) {
          sectionRecommendations.push(
            `Students are struggling with the following letters: ${topLetters.join(
              ", "
            )}.`
          );
        }
      }

      recommendations[section] = sectionRecommendations;
    });

    setSectionRecommendations(recommendations);
  };

  // Handle dynamic data based on section selection
  const getChartData = () => {
    if (selectedSection === "All Sections") {
      return {
        labels: Object.keys(averageScores),
        data: Object.values(averageScores),
      };
    } else {
      return {
        labels: [selectedSection],
        data: [averageScores[selectedSection] || 0],
      };
    }
  };

  // Prepare data for the chart
  const { labels: sectionLabel, data: sectionData } = getChartData();

  return (
    <div className="px-9">
      <div className="flex items-center mb-4">
        <h1 className="text-3xl font-semibold">Admin Dashboard</h1>
        <Tooltip
          showArrow={true}
          content={
            <div className="p-2 text-sm">
              <div className="font-bold">Dashboard Info</div>
              <p>View system data for each section.</p>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="ml-2 text-black" />
        </Tooltip>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-[#fb6ea4] rounded-lg shadow p-4 flex flex-row justify-between">
          <div className="text-white">
            <h2 className="text-xl font-semibold">Students</h2>
            <p className="text-5xl mt-2">{students.length}</p>
          </div>
          <div>
            <FontAwesomeIcon
              icon={faChildReaching}
              size="2xl"
              className="text-md"
              inverse
            />
          </div>
        </div>

        <div className="bg-[#7668d2] rounded-lg shadow p-4 flex flex-row justify-between">
          <div className="text-white">
            <h2 className="text-xl font-semibold">Teachers</h2>
            <p className="text-5xl mt-2">{teachersCount}</p>
          </div>
          <div>
            <FontAwesomeIcon
              icon={faChalkboardTeacher}
              size="2xl"
              className="text-md"
              inverse
            />
          </div>
        </div>

        <div className="bg-[#91c123] rounded-lg shadow p-4 flex flex-row justify-between">
          <div className="text-white">
            <h2 className="text-xl font-semibold">Parents</h2>
            <p className="text-5xl mt-2">{parentsCount}</p>
          </div>
          <div>
            <FontAwesomeIcon
              icon={faHouse}
              size="2xl"
              className="text-md"
              inverse
            />
          </div>
        </div>

        <div className="col-span-1 lg:col-span-3 bg-white rounded-lg shadow p-4">
          <h2 className="text-xl font-semibold">Students Per Section</h2>
          <BarChart
            xAxis={[
              {
                label: "Sections",
                scaleType: "band",
                data: sectionLabels,
                tickSize: 10, // Larger tick size for better readability
              },
            ]}
            yAxis={[
              {
                label: "Number of Students",
                tickCount: 6,
                grid: true,
              },
            ]}
            series={[
              {
                data: Object.values(sectionCounts), // Total students per section
                color: "#4b99f5",
                label: "Total Students",
              },
            ]}
            height={300}
          />
        </div>
      </div>

      <div className="mt-6 bg-white p-6 rounded-md shadow">
        <h2 className="text-2xl font-semibold mb-4 items-center flex flex-col">
          Student Status
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-[#ff505b] rounded-lg shadow-md p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Incomplete</h2>
              <p className="text-5xl mt-2">{statusCounts.Incomplete}</p>
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChildReaching}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>

          <div className="bg-[#ff7828] rounded-lg shadow-md p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Low Emerging Reader</h2>
              <p className="text-5xl mt-2">{statusCounts.LowEmergingReader}</p>
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChalkboardTeacher}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>

          <div className="bg-[#DC84F3] rounded-lg shadow-md p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">High Emerging Reader</h2>
              <p className="text-5xl mt-2">{statusCounts.HighEmergingReader}</p>
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChalkboardTeacher}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>

          <div className="bg-[#4b99f5] rounded-lg shadow-md p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Developing Reader</h2>
              <p className="text-5xl mt-2">{statusCounts.DevelopingReader}</p>
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChildReaching}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>

          <div className="bg-[#ffce1f] rounded-lg shadow-md p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Transitioning Reader</h2>
              <p className="text-5xl mt-2">
                {statusCounts.TransitioningReader}
              </p>
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChildReaching}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>

          <div className="bg-[#FF8080] rounded-lg shadow-md p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Grade Level Reader</h2>
              <p className="text-5xl mt-2">{statusCounts.GradeLevelReader}</p>
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChildReaching}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>
        </div>

        <div className="col-span-1 lg:col-span-2 bg-slate-100 rounded-lg shadow-md p-4 mt-6">
          <h2 className="text-xl font-semibold">Student Status Per Section</h2>
          <BarChart
            xAxis={[
              {
                label: "Sections",
                scaleType: "band",
                data: sectionLabels,
                tickSize: 10,
              },
            ]}
            yAxis={[
              {
                label: "Number of Students",
                tickCount: 6,
                grid: true,
                max: 20,
                tickFormat: (value) => Math.floor(value),
              },
            ]}
            series={statusSeries.map((series) => ({
              ...series,
              element: (
                <Tooltip
                  content={
                    sectionRecommendations[series.label] &&
                    sectionRecommendations[series.label].length > 0 ? (
                      <div>
                        {sectionRecommendations[series.label].map(
                          (rec, index) => (
                            <p key={index}>{rec}</p>
                          )
                        )}
                      </div>
                    ) : (
                      "No specific recommendation available"
                    )
                  }
                  placement="top"
                >
                  <div
                    style={{
                      height: "100%",
                      width: "100%",
                      backgroundColor: series.color,
                      cursor: "pointer",
                    }}
                    onClick={() => handleSectionClick(series.label)} // Add click handler
                  ></div>
                </Tooltip>
              ),
            }))}
            gap={20}
            height={500}
          />
        </div>

        <div className="col-span-1 lg:col-span-2 bg-slate-100 rounded-lg shadow-md p-4 mt-6">
          {/* Dropdown for section selection */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-xl font-semibold mb-4">
              Select Section for Recommendations
            </h2>
            <Select
              label="Select Section"
              variant="bordered"
              className="max-w-xs"
              selectedKeys={[selectedSection]}
              onChange={(e) => setSelectedSection(e.target.value)}
            >
              <SelectItem key="All Sections">All Sections</SelectItem>
              {sections.map((section) => (
                <SelectItem key={section}>{section}</SelectItem>
              ))}
            </Select>
          </div>

          {/* Recommendations */}
          <div className="col-span-1 lg:col-span-2 bg-slate-100 rounded-lg shadow-md p-4 mt-6">
            <h2 className="text-xl font-semibold mb-4">
              Recommendations for {selectedSection}
            </h2>
            {selectedSection === "All Sections" ? (
              <div>
                <p className="text-sm text-gray-700">
                  Initial recommendations for all sections:
                </p>
                <ul className="list-disc list-inside space-y-2">
                  {Object.keys(sectionRecommendations).map((section) =>
                    sectionRecommendations[section].map((rec, index) => (
                      <li key={index} className="text-sm text-gray-700">
                        <strong>{section}:</strong> {rec}
                      </li>
                    ))
                  )}
                </ul>
              </div>
            ) : (
              <div>
                {sectionRecommendations[selectedSection] &&
                sectionRecommendations[selectedSection].length > 0 ? (
                  <ul className="list-disc list-inside space-y-2">
                    {sectionRecommendations[selectedSection].map(
                      (rec, index) => (
                        <li key={index} className="text-sm text-gray-700">
                          {rec}
                        </li>
                      )
                    )}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500">
                    No specific recommendations available for this section.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6 bg-white p-6 rounded-md shadow">
        <h2 className="text-2xl font-semibold mb-4 items-center flex flex-col">
          Student Assessment and Performance
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-[#7668d2] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Assessment</h2>
              <p className="text-5xl mt-2">{assessments.length}</p>{" "}
              {/* Corrected */}
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChalkboardTeacher}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
          </div>

          <div className="bg-[#91c123] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
            <div className="text-white">
              <h2 className="text-xl font-semibold">Performance</h2>
              <p className="text-5xl mt-2">{performanceCounts.length}</p>{" "}
              {/* Corrected */}
            </div>
            <div>
              <FontAwesomeIcon
                icon={faChildReaching}
                size="2xl"
                className="text-md"
                inverse
              />
            </div>
            s
          </div>

          <div className="col-span-1 lg:col-span-3 bg-slate-100 rounded-lg shadow-md p-4 mt-6">
            <h2 className="text-xl font-semibold">Assessment Created</h2>
            <BarChart
              xAxis={[
                {
                  label: "Activities",
                  scaleType: "band",
                  data: Object.keys(activityCounts),
                  tickSize: 10, // Larger tick size for better readability
                },
              ]}
              yAxis={[
                {
                  label: "Number of Assessments",
                  max: 10,
                  min: 0,
                  tickCount: 6, // Set tick count to control number of ticks on the Y-axis
                  tickFormat: (value) => Math.floor(value), // Ensure the ticks are integers
                  grid: true, // Add gridlines for better readability
                },
              ]}
              series={[
                {
                  data: Object.values(activityCounts),
                  color: "#7668d2", // Primary color for bars
                  label: "Assessments", // Label for better clarity
                },
              ]}
              barsize={40} // Increase bar size for better visuals
              gap={30} // Increase gap between bars for a cleaner look
              height={500}
            />
          </div>

          <div className="col-span-1 lg:col-span-3 bg-slate-100 rounded-lg shadow-md p-4 mt-6">
            {/* Dropdown or Buttons to Select Section */}
            <div className="flex justify-start gap-4 mb-4">
              <div>
                <h2 className="text-2xl font-semibold">Select Section:</h2>
                <Select
                  variant="bordered"
                  className="bg-white rounded-sm"
                  placeholder="All Sections"
                  value={selectedSection}
                  onChange={(e) => setSelectedSection(e.target.value)}
                >
                  <SelectItem key="All Sections">All Sections</SelectItem>
                  {sections.map((section) => (
                    <SelectItem key={section}>{section}</SelectItem>
                  ))}
                </Select>
              </div>

              {/* Dropdown to Select Assessment */}
              <div>
                <h2 className="text-2xl font-semibold">Select Assessment:</h2>
                <Select
                  variant="bordered"
                  className="bg-white rounded-sm"
                  placeholder="Select Assessment"
                  value={selectedAssessment}
                  onChange={(e) => setSelectedAssessment(e.target.value)}
                >
                  <SelectItem key="Pagbabaybay">Assessment 1</SelectItem>
                  <SelectItem key="Pantig">Assessment 2</SelectItem>
                  <SelectItem key="Salita">Assessment 3</SelectItem>
                  <SelectItem key="Pagbabasa">Assessment 4</SelectItem>
                </Select>
              </div>
            </div>

            <h2 className="text-xl font-semibold">
              Average Scores in {selectedAssessment} Assessment
            </h2>
            <BarChart
              xAxis={[
                {
                  label: "Sections",
                  scaleType: "band",
                  data: sectionLabel,
                  tickSize: 10,
                },
              ]}
              yAxis={[
                {
                  label: "Average Score",
                  min: 0,
                  max: 10, // Assuming the max score is 100
                  tickCount: 6,
                  tickFormat: (value) => Math.floor(value),
                },
              ]}
              series={[
                {
                  data: sectionData, // Average scores per section or per selected section
                  color: "#91c123",
                  label: "Average Score",
                },
              ]}
              height={500}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default BodyAdminDashboard;
