import { useState, useEffect, useContext } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faChalkboardTeacher,
  faChildReaching,
  faCircleInfo,
} from "@fortawesome/free-solid-svg-icons";
import { Tooltip } from "@nextui-org/react";
import { BarChart } from "@mui/x-charts";
import axios from "axios";
import { UserContext } from "../../../context/userContext";

const BodyTeacherDashboard = () => {
  const [teacher, setTeacher] = useState({});
  const [students, setStudents] = useState([]);
  const [assessments, setAssessments] = useState([]);
  const { user } = useContext(UserContext);

  const [sectionCounts, setSectionCounts] = useState({});
  const [activityCounts, setActivityCounts] = useState({});
  const [performanceCounts, setPerformanceCounts] = useState([]);

  const [statusCounts, setStatusCounts] = useState({
    Incomplete: 0,
    LowEmergingReader: 0,
    HighEmergingReader: 0,
    DevelopingReader: 0,
    TransitioningReader: 0,
    GradeLevelReader: 0,
  });

  useEffect(() => {
    const fetchTeacher = async () => {
      try {
        const response = await axios.get(`/api/getTeacher/${user.UserID}`);
        setTeacher(response.data);
      } catch (err) {
        console.error("Error fetching teacher:", err);
      }
    };

    const fetchStudents = async () => {
      try {
        const response = await axios.get("/api/getStudents");
        const data = response.data;

        const filteredStudents = data.filter(
          (student) => student.Section === teacher.Section
        );
        setStudents(filteredStudents);

        // Group students by their status and update the state
        const statusGroups = {
          Incomplete: 0,
          LowEmergingReader: 0,
          HighEmergingReader: 0,
          DevelopingReader: 0,
          TransitioningReader: 0,
          GradeLevelReader: 0,
        };

        filteredStudents.forEach((student) => {
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
        setSectionCounts(filteredStudents);
      } catch (err) {
        console.error("Error fetching students:", err);
      }
    };

    const fetchPerformance = async () => {
      try {
        const response = await axios.get("/api/getPerformance");
        const data = response.data;

        const filteredPerformance = data.filter(
          (performance) => performance.Section === teacher.Section
        );
        setPerformanceCounts(filteredPerformance);
      } catch (err) {
        console.error("Error fetching performance:", err);
      }
    };

    const fetchAssessments = async () => {
      try {
        const response = await axios.get("/api/getAssessments");
        const data = response.data;

        const filteredAssessments = data.filter(
          (assessment) => assessment.Section === teacher.Section
        );

        const activities = ["Pagbabaybay", "Pantig", "Salita", "Pagbabasa"];
        const counts = activities.reduce((acc, activity) => {
          acc[activity] = filteredAssessments.filter(
            (type) => type.Type === activity
          ).length;
          return acc;
        }, {});

        setAssessments(filteredAssessments);
        setActivityCounts(counts);
      } catch (err) {
        console.error("Error fetching assessments:", err);
      }
    };

    fetchTeacher();
    fetchStudents();
    fetchPerformance();
    fetchAssessments();
  }, [user.UserID, teacher.Section]);

  const assessmentTypes = ["Pagbabaybay", "Pantig", "Salita", "Pagbabasa"];
  const maxScores = { Pagbabaybay: 10, Pantig: 10, Salita: 10, Pagbabasa: 5 };

  const studentsPerformanceData = students.map((student) => {
    const studentPerformances = performanceCounts.filter(
      (performance) => performance.LRN === student.LRN
    );

    const scoresByAssessment = assessmentTypes.map((assessment) => {
      const studentAssessment = studentPerformances.find(
        (perf) => perf.Type === assessment
      );
      return studentAssessment ? parseInt(studentAssessment.Score, 10) : 0;
    });

    return {
      studentName: student.Name,
      scores: scoresByAssessment,
    };
  });

  const assessmentLabels = assessments.map((assessment) => assessment.Type); // Dynamic assessment labels

  return (
    <div className="p-6">
      <div className="flex items-center mb-6">
        <h1 className="text-3xl font-semibold">Teacher Dashboard</h1>
        <Tooltip
          showArrow={true}
          content={
            <div className="p-2 text-sm">
              <div className="font-bold">Dashboard Info</div>
              <p>View system data for the assigned section.</p>
            </div>
          }
        >
          <FontAwesomeIcon icon={faCircleInfo} className="ml-2 text-black" />
        </Tooltip>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-[#fb6ea4] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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
        </div>
      </div>

      <div className="mt-6 bg-white p-6 rounded-md shadow">
        <h2 className="text-2xl font-semibold mb-4 items-center flex flex-col">
          Student Status
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-[#ff505b] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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

          <div className="bg-[#ff7828] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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

          <div className="bg-[#DC84F3] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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

          <div className="bg-[#4b99f5] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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

          <div className="bg-[#ffce1f] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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

          <div className="bg-[#FF8080] rounded-lg shadow-2xl p-4 flex flex-row justify-between">
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
      </div>

      <div className="mt-6">
        <h2 className="text-2xl font-semibold mb-4">Assessment Created</h2>
        <div className="bg-white rounded-lg shadow-2xl p-4">
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
                color: "#ffce1f", // Primary color for bars
                label: "Assessments", // Label for better clarity
              },
            ]}
            barsize={40} // Increase bar size for better visuals
            gap={30} // Increase gap between bars for a cleaner look
            width={1500}
            height={400}
          />
        </div>
      </div>

      {/* Performance of students in a dynamic bar chart */}
      <div className="mt-6">
        <h2 className="text-2xl font-semibold mb-4">Performance of Students</h2>
        <div className="bg-white rounded-lg shadow-2xl p-4">
          {performanceCounts.length > 0 && students.length > 0 ? (
            <BarChart
              xAxis={[
                {
                  label: "Assessment Type",
                  data: assessmentTypes,
                  scaleType: "band",
                  tickSize: 10,
                },
              ]}
              yAxis={[
                {
                  label: "Score",
                  min: 0,
                  max: 10, // Set max score to 10 since it's the highest among assessments
                  tickCount: 6,
                  tickFormat: (value) => Math.floor(value),
                },
              ]}
              series={studentsPerformanceData.map((studentData) => ({
                data: studentData.scores,
                label: studentData.studentName,
                color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
              }))}
              width={1500}
              height={500}
            />
          ) : (
            <p>No performance data available</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default BodyTeacherDashboard;
