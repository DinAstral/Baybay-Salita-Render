import { useState } from "react";
import {
  Button,
  Card,
  CardBody,
  Checkbox,
  CheckboxGroup,
} from "@nextui-org/react";
import { Link, useNavigate } from "react-router-dom";

const RegisterSelect = () => {
  const [selectedRoles, setSelectedRoles] = useState([]);
  const [isInvalid, setIsInvalid] = useState(true);
  const navigate = useNavigate();

  const loginButton = () => {
    navigate("/login");
  };

  const handleNext = () => {
    if (selectedRoles.length === 1) {
      const selectedRole = selectedRoles[0];
      if (selectedRole === "Parent") {
        navigate("/registerParent");
      } else if (selectedRole === "Teacher") {
        navigate("/registerTeacher");
      }
    } else {
      setIsInvalid(true);
    }
  };

  return (
    <div className="flex bg-[#f6fbff] w-full h-screen items-center justify-center p-4">
      <div className="w-full max-w-lg flex flex-col items-center justify-center">
        <Card className="w-full p-6 sm:p-8">
          <h1 className="text-2xl sm:text-3xl font-semibold text-center mb-4 text-gray-700">
            Registration
          </h1>
          <p className="text-sm text-center mb-6">
            Please fill up the details needed!
          </p>
          <CardBody>
            <form>
              <div className="flex flex-col items-center justify-center mb-6">
                <CheckboxGroup
                  value={selectedRoles}
                  onChange={(value) => {
                    setSelectedRoles(value);
                    setIsInvalid(value.length !== 1);
                  }}
                  orientation="horizontal"
                  isInvalid={isInvalid}
                  className="space-x-4"
                >
                  <Checkbox value="Parent">Parent</Checkbox>
                  <Checkbox value="Teacher">Teacher</Checkbox>
                </CheckboxGroup>
                {isInvalid && (
                  <p className="text-red-500 text-sm mt-2">
                    Please select exactly one role.
                  </p>
                )}
              </div>
              <div className="flex items-center justify-center gap-4">
                <Button
                  className="my-2"
                  size="lg"
                  radius="md"
                  color="danger"
                  variant="light"
                  onClick={loginButton}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  className="my-2"
                  size="lg"
                  radius="md"
                  color="primary"
                  onClick={handleNext}
                  disabled={isInvalid}
                >
                  Next
                </Button>
              </div>
            </form>
          </CardBody>
        </Card>
      </div>
    </div>
  );
};

export default RegisterSelect;
