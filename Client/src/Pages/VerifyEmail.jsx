import { useState, useCallback } from "react";
import axios from "axios";
import { toast } from "react-hot-toast";
import { Input, Button, Card, CardBody } from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Link, useNavigate } from "react-router-dom";
import { faLock } from "@fortawesome/free-solid-svg-icons";

const VerifyEmail = ({ verificationType = "email" }) => {
  const navigate = useNavigate();
  const [otp, setOtp] = useState("");
  const [loading, setLoading] = useState(false);

  const verifyUser = async (e) => {
    e.preventDefault();
    const userId = localStorage.getItem("userId");

    if (!userId) {
      toast.error("User ID is missing. Please try registering again.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post("/api/verify", { userId, otp });

      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setOtp("");
        toast.success("Verification successful. Please log in.");
        navigate("/login");
      }
    } catch (error) {
      console.error(error);
      toast.error("An error occurred during verification. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex bg-[#f6fbff] w-full h-screen justify-center items-center">
      <div className="w-full max-w-md px-6 sm:px-8 lg:px-12">
        <Card className="p-6">
          <h1 className="text-2xl lg:text-3xl font-bold text-center">
            Verify {verificationType === "email" ? "Email" : "Phone"}
          </h1>
          <p className="text-sm text-center mt-2">
            Please enter the OTP code sent to your {verificationType} for
            verification.
          </p>
          <CardBody>
            <form onSubmit={verifyUser}>
              <div className="flex flex-col gap-4">
                <Input
                  type="text"
                  label="Enter the code"
                  variant="bordered"
                  className="bg-transparent py-1 my-1"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  required
                  endContent={
                    <FontAwesomeIcon
                      icon={faLock}
                      className="text-xl text-default-400 flex-shrink-0"
                    />
                  }
                />
              </div>

              {/* Submit and Back to Login buttons */}
              <div className="w-full flex items-center justify-center gap-6 mt-6">
                <Button
                  type="submit"
                  className="w-full sm:w-auto"
                  size="lg"
                  radius="md"
                  color="primary"
                  disabled={loading}
                >
                  {loading ? "Verifying..." : "Submit"}
                </Button>
              </div>
            </form>
          </CardBody>
        </Card>
      </div>
    </div>
  );
};

export default VerifyEmail;
