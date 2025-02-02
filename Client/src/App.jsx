import React, { Suspense } from "react";
import { NextUIProvider } from "@nextui-org/react";
import { Route, Routes, useNavigate } from "react-router-dom";
import axios from "axios";
import { Toaster } from "react-hot-toast";
import { UserContextProvider } from "../context/userContext";
import RoleBasedRoute from "../hooks/RoleBasedRoute";
import Loader from "./Components/Loader/Loader";

// Lazy load components
const Login = React.lazy(() => import("./Pages/Login"));
const RegisterSelect = React.lazy(() => import("./Pages/RegisterSelect"));
const ForgotPassword = React.lazy(() => import("./Pages/ForgotPassword"));
const VerifyEmail = React.lazy(() => import("./Pages/VerifyEmail"));
const AdminDashboard = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminDashboard")
);
const AdminEditProfile = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminEditProfile")
);
const AdminUser = React.lazy(() => import("./Pages/AdminDashboard/AdminUser"));
const AdminEditUser = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminEditUser")
);
const AdminStudent = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminStudent")
);
const AdminViewStudent = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminViewStudent")
);
const AdminAssessment = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminAssessment")
);
const AdminEditStudent = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminEditStudent")
);
const AdminViewParent = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminViewParent")
);
const AdminEditParent = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminEditParent")
);
const AdminAddUser = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminAddUser")
);
const AdminViewTeacher = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminViewTeacher")
);
const AdminEditTeacher = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminEditTeacher")
);
const AdminProfile = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminProfile")
);
const AdminAddStudent = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminAddStudent")
);
const TeacherRegister = React.lazy(() => import("./Pages/TeacherRegister"));
const ParentRegister = React.lazy(() => import("./Pages/ParentRegister"));
const Main = React.lazy(() => import("./Pages/Home/Main"));
const ResetPassword = React.lazy(() => import("./Pages/ResetPassword"));
const Dashboard = React.lazy(() =>
  import("./Pages/TeacherDashboard/Dashboard")
);
const TeacherProfile = React.lazy(() =>
  import("./Pages/TeacherDashboard/TeacherProfile")
);
const ManageStudent = React.lazy(() =>
  import("./Pages/TeacherDashboard/ManageStudent")
);
const AddStudent = React.lazy(() =>
  import("./Pages/TeacherDashboard/AddStudent")
);
const EditStudent = React.lazy(() =>
  import("./Pages/TeacherDashboard/EditStudent")
);
const ViewAssessment = React.lazy(() =>
  import("./Pages/TeacherDashboard/ViewAssessment")
);
const ViewStudent = React.lazy(() =>
  import("./Pages/TeacherDashboard/ViewStudent")
);
const UnauthorizedPage = React.lazy(() => import("./Pages/UnauthorizedPage"));
const InformationKid = React.lazy(() =>
  import("./Pages/ParentDashboard/InformationKid")
);
const ParentFeedbackTeacher = React.lazy(() =>
  import("./Pages/ParentDashboard/ParentFeedbackTeacher")
);
const ParentViewStudentProgress = React.lazy(() =>
  import("./Pages/ParentDashboard/ParentViewStudentProgress")
);
const ParentProfile = React.lazy(() =>
  import("./Pages/ParentDashboard/ParentProfile")
);
const ProfileUpdate = React.lazy(() =>
  import("./Pages/ParentDashboard/ProfileUpdate")
);
const AdminViewPerformance = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminViewPerformance")
);
const TestPage = React.lazy(() => import("./Pages/testpage"));
const AdminViewAssessment = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminViewAssessment")
);
const AdminViewStudentPerformance = React.lazy(() =>
  import("./Pages/AdminDashboard/AdminViewStudentPerformance")
);
const ManagePerformance = React.lazy(() =>
  import("./Pages/TeacherDashboard/ManagePerformance")
);
const ViewStudentAssessment = React.lazy(() =>
  import("./Pages/TeacherDashboard/ViewStudentAssessment")
);
const ViewStudentPerformance = React.lazy(() =>
  import("./Pages/TeacherDashboard/ViewStudentPerformance")
);

// Configure axios base URL
axios.defaults.baseURL = "https://baybay-salita-heroku.onrender.com";
axios.defaults.withCredentials = true;

const App = () => {
  const navigate = useNavigate();
  return (
    <div>
      {/* User Credentials */}
      <UserContextProvider>
        {/* UI */}
        <NextUIProvider navigate={navigate}>
          {/* Notification */}
          <Toaster position="top-center" toastOptions={{ duration: 3000 }} />

          {/* Routes */}
          <Suspense fallback={<Loader />}>
            <Routes>
              <Route path="/" element={<Main />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<RegisterSelect />} />
              <Route path="/registerTeacher" element={<TeacherRegister />} />
              <Route path="/registerParent" element={<ParentRegister />} />
              <Route path="/forgotPassword" element={<ForgotPassword />} />
              <Route path="/test" element={<TestPage />} />
              <Route
                path="/api/reset-password/:id/:token"
                element={<ResetPassword />}
              />
              <Route path="/verifyEmail" element={<VerifyEmail />} />
              <Route path="/unauthorized" element={<UnauthorizedPage />} />

              {/* Teacher Dashboard Routes */}
              <Route
                path="/teacherDashboard"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <Dashboard />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/teacherProfile"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <TeacherProfile />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/manageStudent"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <ManageStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/addStudent"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <AddStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/viewStudent/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <ViewStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/editStudent/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <EditStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/viewAssessment"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <ViewAssessment />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/managePerformance"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <ManagePerformance />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/ViewActivityAssessment/:ActivityCode"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <ViewStudentAssessment />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/ViewStudentPerformance/:UserInputId"
                element={
                  <RoleBasedRoute allowedRoles={["Teacher"]}>
                    <ViewStudentPerformance />
                  </RoleBasedRoute>
                }
              />

              {/* Parent Dashboard Routes */}
              <Route
                path="/parentDashboard"
                element={
                  <RoleBasedRoute allowedRoles={["Parent"]}>
                    <InformationKid />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/parentFeedbackTeacher"
                element={
                  <RoleBasedRoute allowedRoles={["Parent"]}>
                    <ParentFeedbackTeacher />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/parentStudentProgress"
                element={
                  <RoleBasedRoute allowedRoles={["Parent"]}>
                    <ParentViewStudentProgress />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/parentProfile"
                element={
                  <RoleBasedRoute allowedRoles={["Parent"]}>
                    <ParentProfile />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/parentUpdateProfile/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Parent"]}>
                    <ProfileUpdate />
                  </RoleBasedRoute>
                }
              />

              {/* Admin Dashboard Routes */}
              <Route
                path="/AdminDashboard"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminDashboard />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/AdminEditProfile/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminEditProfile />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/AdminUsers"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminUser />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/AdminEditUser/:UserID"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminEditUser />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminStudents"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminAddStudent"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminAddStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminViewStudent/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminViewStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminStudentAssessment"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminAssessment />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminViewPerformance"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminViewPerformance />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminEditStudent/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminEditStudent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminViewParent/:UserID"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminViewParent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminEditParent/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminEditParent />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminAddUser"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminAddUser />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminViewTeacher/:UserID"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminViewTeacher />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminEditTeacher/:id"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminEditTeacher />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/adminProfile"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminProfile />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/AdminViewAssessment/:ActivityCode"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminViewAssessment />
                  </RoleBasedRoute>
                }
              />
              <Route
                path="/AdminViewStudentPerformance/:UserInputId"
                element={
                  <RoleBasedRoute allowedRoles={["Admin"]}>
                    <AdminViewStudentPerformance />
                  </RoleBasedRoute>
                }
              />
            </Routes>
          </Suspense>
        </NextUIProvider>
      </UserContextProvider>
    </div>
  );
};

export default App;
