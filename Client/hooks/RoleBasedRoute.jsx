/* eslint-disable react/prop-types */
import { useContext } from "react";
import { Navigate } from "react-router-dom";
import { UserContext } from "../context/userContext"; // Adjust the import path as needed
import Loader from "../src/Components/Loader/Loader";

const RoleBasedRoute = ({ children, allowedRoles }) => {
  const { user, loading } = useContext(UserContext);

  // If still loading user data, you may want to show a loader or nothing
  if (loading) {
    return <Loader />; // Show loading state or nothing
  }

  // If no user data is present, redirect to the login page
  if (!user) {
    return <Navigate to="/" />;
  }

  // If user's role is not allowed, redirect to unauthorized page
  if (!allowedRoles.includes(user.role)) {
    return <Navigate to="/unauthorized" />;
  }

  // Render the children if all checks pass
  return children;
};

export default RoleBasedRoute;
