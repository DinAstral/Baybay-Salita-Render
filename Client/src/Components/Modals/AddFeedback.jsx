/* eslint-disable react/prop-types */
import { useState } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import PropTypes from "prop-types";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Input,
  Textarea,
  Select,
  SelectItem,
} from "@nextui-org/react";

const AddSuccess = ({ show, onHide }) => {
  const handleSuccessClick = () => {
    onHide();
    window.location.reload(); // You might want to change this to a more efficient state update instead of reloading the page.
  };

  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="success-modal-title"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader id="success-modal-title">
          Feedback Created Successfully!
        </ModalHeader>
        <ModalBody>
          <p>You have created feedback for your student's performance!</p>
        </ModalBody>
        <ModalFooter>
          <Button color="primary" onClick={handleSuccessClick}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

const AddFeedback = ({ show, onHide, actCode, userid, lrn, section }) => {
  const [modalShow, setModalShow] = useState(false);
  const [data, setData] = useState({
    Title: "",
    Type: "",
    Feedback_Date: "",
    Context: "",
  });

  const [errors, setErrors] = useState({});

  // Validation function
  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    if (!data.Title) {
      newErrors.Title = "Title is required.";
      isValid = false;
    }
    if (!data.Type) {
      newErrors.Type = "Type of assessment is required.";
      isValid = false;
    }
    if (!data.Feedback_Date) {
      newErrors.Feedback_Date = "Feedback date is required.";
      isValid = false;
    }
    if (!data.Context) {
      newErrors.Context = "Context is required.";
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const createFeedback = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out all required fields.");
      return;
    }

    const { Feedback_Date, Type, Title, Context } = data;

    try {
      const response = await axios.post(`/api/submitFeedback`, {
        UserID: userid,
        LRN: lrn,
        Section: section,
        Title,
        ActivityCode: actCode,
        Type,
        Feedback_Date,
        Context,
      });
      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({
          Title: "",
          Type: "",
          Feedback_Date: "",
          Context: "",
        });
        toast.success("Created Feedback Successfully.");
        setModalShow(true);
      }
    } catch (error) {
      console.error(error);
      toast.error("Error creating feedback. Please try again.");
    }
  };

  return (
    <>
      <Modal
        isOpen={show}
        onClose={onHide}
        size="lg"
        aria-labelledby="create-feedback-modal-title"
        isDismissable={false}
        isKeyboardDismissDisabled={true}
        scrollBehavior={"inside"}
      >
        <ModalContent>
          <ModalHeader id="create-feedback-modal-title">
            Create New Feedback
          </ModalHeader>
          <form onSubmit={createFeedback}>
            <ModalBody>
              <Input
                type="text"
                label="Title"
                placeholder="Input a concern title"
                labelPlacement="outside"
                variant="bordered"
                className="bg-transparent py-1 my-1"
                value={data.Title}
                onChange={(e) => setData({ ...data, Title: e.target.value })}
                isInvalid={!!errors.Title}
                errorMessage={errors.Title}
              />

              <Select
                labelPlacement="outside"
                label="Type of Assessment"
                placeholder="Select Type of Assessment"
                aria-label="Select type of assessment"
                variant="bordered"
                className="bg-transparent py-1 my-1"
                value={data.Type}
                onChange={(e) => setData({ ...data, Type: e.target.value })}
                isInvalid={!!errors.Type}
                errorMessage={errors.Type}
              >
                <SelectItem key="">Select Type of Assessment</SelectItem>
                <SelectItem key="Pagbabaybay">
                  Assessment 1: Pagbabaybay Tunog at Letra
                </SelectItem>
                <SelectItem key="Pantig">Assessment 2: Pantig</SelectItem>
                <SelectItem key="Salita">Assessment 3: Salita</SelectItem>
                <SelectItem key="Pagbabasa">Assessment 4: Pagbabasa</SelectItem>
              </Select>

              <Input
                className="pt-2"
                type="date"
                label="Feedback Date"
                labelPlacement="outside"
                variant="bordered"
                value={data.Feedback_Date}
                onChange={(e) =>
                  setData({ ...data, Feedback_Date: e.target.value })
                }
                isInvalid={!!errors.Feedback_Date}
                errorMessage={errors.Feedback_Date}
              />

              <Textarea
                placeholder="Input feedback context"
                label="Context"
                labelPlacement="outside"
                variant="bordered"
                className="bg-transparent py-1 my-1"
                value={data.Context}
                onChange={(e) => setData({ ...data, Context: e.target.value })}
                isInvalid={!!errors.Context}
                errorMessage={errors.Context}
              />
            </ModalBody>
            <ModalFooter>
              <Button color="danger" variant="light" onClick={onHide}>
                Close
              </Button>
              <Button color="primary" type="submit">
                Send Feedback
              </Button>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
      <AddSuccess show={modalShow} onHide={() => setModalShow(false)} />
    </>
  );
};

// Add prop types validation
AddFeedback.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  actCode: PropTypes.string.isRequired,
  userid: PropTypes.string.isRequired,
  lrn: PropTypes.string.isRequired,
  section: PropTypes.string.isRequired,
};

AddSuccess.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
};

export default AddFeedback;
