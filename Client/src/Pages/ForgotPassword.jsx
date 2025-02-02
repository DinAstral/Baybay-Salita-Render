import { useState } from "react";
import { Input, Button, Card, CardBody } from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Link, useNavigate } from "react-router-dom";
import { faEnvelope } from "@fortawesome/free-solid-svg-icons";

import { toast } from "react-hot-toast";
import axios from "axios";

const ForgotPassword = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false); // Loading state
  const [errors, setErrors] = useState({}); // Validation errors

  const loginButton = () => {
    navigate("/login");
  };

  const [data, setData] = useState({
    email: "",
  });

  // Validate email
  const validateEmail = (email) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

  const forgotPassword = async (e) => {
    e.preventDefault();

    // Clear previous errors
    setErrors({});

    // Check if email is valid
    if (!validateEmail(data.email)) {
      setErrors({ email: "Please enter a valid email address." });
      return;
    }

    setLoading(true); // Start loading
    try {
      const response = await axios.post("/api/forgotPass", {
        email: data.email,
      });

      const responseData = response.data;
      if (responseData.error) {
        toast.error(responseData.error);
      } else {
        setData({
          email: "",
        });
        toast.success("We have e-mailed your password reset link!");
        navigate("/login");
      }
    } catch (error) {
      console.log(error);
      toast.error("Error occurred while processing the request.");
    } finally {
      setLoading(false); // End loading
    }
  };

  return (
    <div className="flex bg-[#f6fbff] w-full h-screen items-center justify-center p-4">
      <div className="w-full max-w-md">
        <Card className="w-full flex flex-col p-6">
          <h1 className="text-2xl md:text-3xl font-bold text-center">
            Forgot Password
          </h1>
          <p className="text-sm md:text-base text-center mt-2">
            Enter your registered email to reset your password.
          </p>
          <CardBody>
            <form onSubmit={forgotPassword}>
              <div className="flex flex-col gap-2">
                <Input
                  type="text"
                  label="Email"
                  variant="bordered"
                  className="bg-transparent py-2 my-1"
                  value={data.email}
                  onChange={(e) => setData({ ...data, email: e.target.value })}
                  endContent={
                    <FontAwesomeIcon
                      icon={faEnvelope}
                      className="text-xl text-default-400 pointer-events-none"
                    />
                  }
                  aria-describedby={errors.email ? "email-error" : null}
                  aria-invalid={errors.email ? true : false}
                />
                {errors.email && (
                  <span id="email-error" className="text-red-500 text-sm">
                    {errors.email}
                  </span>
                )}
              </div>
              <div className="w-full flex items-center justify-center gap-4 my-4">
                <Button
                  className="my-2"
                  size="lg"
                  radius="md"
                  color="danger"
                  variant="light"
                  onClick={loginButton}
                  disabled={loading}
                >
                  Back
                </Button>
                <Button
                  type="submit"
                  className="my-2"
                  size="lg"
                  radius="md"
                  color="primary"
                  disabled={loading}
                >
                  {loading ? "Submitting..." : "Submit"}
                </Button>
              </div>
            </form>
          </CardBody>
        </Card>
      </div>
    </div>
  );
};

export default ForgotPassword;
