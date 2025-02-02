import React, { useState, useContext } from "react";
import { UserContext } from "../../context/userContext";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "react-hot-toast";
import { Input, Checkbox, Button } from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faEnvelope,
  faEyeSlash,
  faEye,
} from "@fortawesome/free-solid-svg-icons";

import Image1 from "../assets/Login_baybay.png";
import Image2 from "../assets/login_blue.png";

const Login = () => {
  const { setUser } = useContext(UserContext);
  const [isVisible, setIsVisible] = useState(false);
  const navigate = useNavigate();
  const [data, setData] = useState({
    email: "",
    password: "",
  });

  const toggleVisibility = () => setIsVisible(!isVisible);
  const [isSelected, setIsSelected] = React.useState(false);

  const registerButton = () => {
    navigate("/register");
  };
  const loginUser = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("/api/login", {
        email: data.email,
        password: data.password,
      });

      if (response && response.data) {
        const responseData = response.data;

        if (responseData.error) {
          toast.error(responseData.error);
          if (responseData.data && responseData.data.userId) {
            localStorage.setItem("userId", responseData.data.userId);
            navigate("/verify-email");
          }
        } else {
          // Successful login actions
          localStorage.setItem("token", responseData.token);
          localStorage.setItem("user", JSON.stringify(responseData.user));
          setUser(responseData.user);
          toast.success("Login Successful.");

          if (responseData.role === "Parent") {
            navigate("/parentDashboard");
          } else if (responseData.role === "Teacher") {
            navigate("/teacherDashboard");
          } else if (responseData.role === "Admin") {
            navigate("/AdminDashboard");
          }
        }
      }
    } catch (error) {
      console.error("Error in loginUser client:", error);
      if (error.response && error.response.data) {
        // Handle expected errors from the server
        toast.error(error.response.data.error);
      } else {
        // Handle unexpected errors
        toast.error("Unexpected error occurred while logging in.");
      }
    }
  };

  return (
    <div className="flex flex-col md:flex-row w-full h-screen">
      {/* Left Image Section */}
      <div className="w-full md:w-[65%] h-full flex items-center justify-center bg-[#FAFAFA]">
        <img
          src={Image1}
          className="w-full h-full object-fill"
          alt="Learning Platform"
        />
      </div>

      {/* Right Form Section */}
      <div className="relative w-full md:w-[35%] h-full flex items-center justify-center bg-[#FAFAFA]">
        <img
          src={Image2}
          className="absolute inset-0 w-full h-full object-fill opacity-50"
          alt="Background Image"
        />
        <div className="relative z-10 w-full max-w-md bg-white shadow-lg rounded-lg p-8 opacity-90">
          <div className="mb-6">
            <h1 className="text-lg font-semibold text-center text-[#060606] mb-4">
              Taytay Elementary School Baybay Salita
            </h1>
            <div className="text-center mb-6">
              <h3 className="text-2xl font-semibold mb-2">Login</h3>
              <p className="text-sm mb-2">
                Welcome Back! Please enter your details.
              </p>
            </div>

            <form onSubmit={loginUser} className="flex flex-col">
              <Input
                type="text"
                label="Email"
                variant="bordered"
                placeholder="Enter your email"
                className="mb-4"
                errorMessage="Please enter a valid email."
                value={data.email}
                onChange={(e) => setData({ ...data, email: e.target.value })}
                endContent={
                  <FontAwesomeIcon
                    icon={faEnvelope}
                    className="text-2xl text-default-400"
                  />
                }
              />

              <Input
                label="Password"
                placeholder="Enter your password"
                variant="bordered"
                className="mb-4"
                value={data.password}
                onChange={(e) => setData({ ...data, password: e.target.value })}
                endContent={
                  <button
                    className="focus:outline-none"
                    type="button"
                    onClick={toggleVisibility}
                    aria-label="toggle password visibility"
                  >
                    {isVisible ? (
                      <FontAwesomeIcon
                        icon={faEyeSlash}
                        className="text-2xl text-default-400"
                      />
                    ) : (
                      <FontAwesomeIcon
                        icon={faEye}
                        className="text-2xl text-default-400"
                      />
                    )}
                  </button>
                }
                type={isVisible ? "text" : "password"}
              />

              <div className="flex items-center justify-between mb-4">
                <Checkbox
                  className="mr-2"
                  isSelected={isSelected}
                  onValueChange={setIsSelected}
                >
                  <p className="text-sm">Remember me</p>
                </Checkbox>

                <Link
                  to="/forgotPassword"
                  className="text-sm font-medium cursor-pointer underline underline-offset-2"
                >
                  Forgot Password?
                </Link>
              </div>

              <Button
                className="my-2"
                size="lg"
                radius="md"
                color="primary"
                type="submit"
              >
                Login
              </Button>

              <Button
                className="my-2"
                size="lg"
                radius="md"
                onClick={registerButton}
              >
                Register
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
