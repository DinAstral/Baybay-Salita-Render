import React, { useState, useEffect, useContext } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faBookOpenReader,
  faComment,
  faEye,
} from "@fortawesome/free-solid-svg-icons";
import { Link, useLocation, useNavigate } from "react-router-dom";
import logo from "../../assets/Sidebar_Final_cut.png";
import { UserContext } from "../../../context/userContext"; // Import UserContext
import { Button } from "@nextui-org/react";

const ParentSidebar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeIndex, setActiveIndex] = useState(null);
  const { clearCookie } = useContext(UserContext); // Access clearCookie from UserContext

  const handleLogout = () => {
    clearCookie(); // Clear the cookie and user state
    navigate("/"); // Redirect to the Home page
  };

  useEffect(() => {
    const menuItem = [
      "/parentDashboard",
      "/parentStudentProgress",
      "/parentFeedbackTeacher",
    ];
    const index = menuItem.findIndex((item) => item === location.pathname);
    setActiveIndex(index);
  }, [location.pathname]);

  return (
    <>
      <div className="flex flex-col gap-10 px-4 h-11 justify-between">
        <div className="logo-dash pr-1">
          <img
            src={logo}
            alt="Baybay Salita Logo"
            onClick={() => navigate("/")}
            className="cursor-pointer"
            draggable="false"
          />
        </div>

        {/* Separation Line */}
        <hr className="border-gray-600 border-t-1" />

        <div className="flex flex-col gap-3 p-3 mb-6 pt-16">
          {[
            {
              icon: faBookOpenReader,
              text: "Information of Kid",
              link: "/parentDashboard",
            },
            {
              icon: faEye,
              text: "Student Progress",
              link: "/parentStudentProgress",
            },
            {
              icon: faComment,
              text: "Feedback of Teacher",
              link: "/parentFeedbackTeacher",
            },
          ].map((item, index) => (
            <Link
              key={index}
              to={item.link}
              className={`flex items-center p-3 font-semibold text-md rounded-lg transition-all duration-200 ${
                index === activeIndex
                  ? "bg-blue-500 text-white shadow-lg"
                  : "text-gray-700 hover:bg-blue-100 hover:text-blue-700"
              }`}
              onClick={() => setActiveIndex(index)}
            >
              <FontAwesomeIcon icon={item.icon} size="lg" className="mr-3" />
              {item.text}
            </Link>
          ))}
        </div>

        {/* Logout Button */}
        <div className="px-3 pt-[16rem]">
          <Button
            color="error"
            className="w-full text-md font-semibold py-3 rounded-lg shadow-md transition-all duration-150 text-white bg-[#ff505b] hover:bg-red-600"
            onClick={handleLogout}
          >
            Logout
          </Button>
        </div>
      </div>
    </>
  );
};

export default ParentSidebar;
