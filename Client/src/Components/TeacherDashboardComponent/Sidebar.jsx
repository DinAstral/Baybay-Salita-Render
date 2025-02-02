import React, { useState, useEffect, useContext } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faHome,
  faEye,
  faSquarePlus,
  faUser,
} from "@fortawesome/free-solid-svg-icons";
import { Link, useLocation, useNavigate } from "react-router-dom";
import logo from "../../assets/Sidebar_Final_cut.png";
import { UserContext } from "../../../context/userContext"; // Import UserContext
import { Button } from "@nextui-org/react";

const Sidebar = () => {
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
      "/teacherDashboard",
      "/manageStudent",
      "/addStudent",
      "/viewAssessment",
      "/managePerformance",
    ];
    const index = menuItem.findIndex((item) => item === location.pathname);
    setActiveIndex(index);
  }, [location.pathname]);

  const toggleActive = (index) => {
    setActiveIndex(index);
  };

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
            { icon: faHome, text: "Dashboard", link: "/teacherDashboard" },
            { icon: faUser, text: "Manage Student", link: "/manageStudent" },
            { icon: faSquarePlus, text: "Add Student", link: "/addStudent" },
            { icon: faEye, text: "Manage Assessment", link: "/viewAssessment" },
            {
              icon: faEye,
              text: "Manage Performance",
              link: "/managePerformance",
            },
          ].map((item, index) => (
            <Link
              key={index}
              to={item.link}
              className={`flex items-center p-2 font-semibold text-md text-gray-850 hover:bg-blue-100 rounded-md ${
                index === activeIndex ? "bg-blue-400 text-white" : ""
              }`}
            >
              <FontAwesomeIcon icon={item.icon} size="1x" className="mr-2" />
              {item.text}
            </Link>
          ))}
        </div>
        <div className="px-3 pt-[15rem]">
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

export default Sidebar;
