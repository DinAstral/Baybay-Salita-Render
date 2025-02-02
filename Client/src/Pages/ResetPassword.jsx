import { useState, useCallback } from "react";
import { Input, Button, Card, CardBody } from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Link, useNavigate, useParams } from "react-router-dom";
import { faEyeSlash, faEye } from "@fortawesome/free-solid-svg-icons";

import { toast } from "react-hot-toast";
import axios from "axios";

const ResetPassword = () => {
  const navigate = useNavigate();
  const { id, token } = useParams();

  // States for form inputs and password visibility
  const [isVisible, setIsVisible] = useState(false);
  const [isVisibleConfirm, setIsVisibleConfirm] = useState(false);
  const [data, setData] = useState({
    password: "",
    confirmPassword: "",
  });

  // Toggle password visibility
  const toggleVisibility = useCallback(() => setIsVisible((prev) => !prev), []);
  const toggleVisibilityConfirm = useCallback(
    () => setIsVisibleConfirm((prev) => !prev),
    []
  );

  // Handle form data changes
  const handleChange = useCallback((e) => {
    setData((prevData) => ({
      ...prevData,
      [e.target.name]: e.target.value,
    }));
  }, []);

  // Navigate to login
  const loginButton = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  // Form submission logic
  const resetPassword = async (e) => {
    e.preventDefault();
    const { password, confirmPassword } = data;

    // Check if passwords match
    if (password !== confirmPassword) {
      toast.error("Passwords do not match!");
      return;
    }

    // Try resetting the password via API
    try {
      const response = await axios.post(`/api/reset-password/${id}/${token}`, {
        password,
      });
      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({ password: "", confirmPassword: "" });
        toast.success("Password reset successfully.");
        navigate("/login");
      }
    } catch (error) {
      toast.error("Something went wrong. Please try again.");
      console.error(error);
    }
  };

  return (
    <div className="flex bg-[#f6fbff] w-full h-screen justify-center items-center">
      <div className="w-full max-w-lg px-6 sm:px-8 lg:px-12">
        <Card className="flex flex-col p-6">
          <h1 className="text-2xl lg:text-3xl font-bold text-center">
            Reset Password
          </h1>
          <p className="text-sm text-center mt-2">
            Enter your new password below.
          </p>
          <CardBody>
            <form onSubmit={resetPassword}>
              <div className="flex flex-col gap-4">
                {/* Password input */}
                <Input
                  name="password"
                  label="Password"
                  variant="bordered"
                  value={data.password}
                  onChange={handleChange}
                  required
                  endContent={
                    <button
                      className="focus:outline-none"
                      type="button"
                      onClick={toggleVisibility}
                      aria-label="Toggle password visibility"
                    >
                      <FontAwesomeIcon
                        icon={isVisible ? faEyeSlash : faEye}
                        className="text-xl"
                      />
                    </button>
                  }
                  type={isVisible ? "text" : "password"}
                />

                {/* Confirm Password input */}
                <Input
                  name="confirmPassword"
                  label="Confirm Password"
                  variant="bordered"
                  value={data.confirmPassword}
                  onChange={handleChange}
                  required
                  endContent={
                    <button
                      className="focus:outline-none"
                      type="button"
                      onClick={toggleVisibilityConfirm}
                      aria-label="Toggle confirm password visibility"
                    >
                      <FontAwesomeIcon
                        icon={isVisibleConfirm ? faEyeSlash : faEye}
                        className="text-xl"
                      />
                    </button>
                  }
                  type={isVisibleConfirm ? "text" : "password"}
                />
              </div>

              {/* Action buttons */}
              <div className="flex justify-between gap-4 mt-6">
                <Button
                  className="w-full sm:w-auto"
                  size="lg"
                  radius="md"
                  color="danger"
                  variant="light"
                  onClick={loginButton}
                >
                  Back to Login
                </Button>
                <Button
                  type="submit"
                  className="w-full sm:w-auto"
                  size="lg"
                  radius="md"
                  color="primary"
                >
                  Submit
                </Button>
              </div>
            </form>
          </CardBody>
        </Card>
      </div>
    </div>
  );
};

export default ResetPassword;
