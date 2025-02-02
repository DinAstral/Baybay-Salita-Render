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
  Select,
  Input,
  SelectItem,
  Textarea,
} from "@nextui-org/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faBook,
  faComment,
  faPen,
  faQuestion,
} from "@fortawesome/free-solid-svg-icons";

const ImportSentenceSuccess = ({ show, onHide }) => {
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
          Sentence Imported Successfully!
        </ModalHeader>
        <ModalBody>
          <p>You have successfully imported a sentence for your students!</p>
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

const ImportSentence = ({ show, onHide }) => {
  const [modalShow, setModalShow] = useState(false);
  const [data, setData] = useState({
    Type: "",
    Title: "",
    Sentence: "",
    Question1: "",
    Question2: "",
    Question3: "",
    Question4: "",
    Question5: "",
    Answer1: "",
    Answer2: "",
    Answer3: "",
    Answer4: "",
    Answer5: "",
  });

  const [errors, setErrors] = useState({});

  // Validation function
  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    if (!data.Type) {
      newErrors.Type = "Type is required.";
      isValid = false;
    }
    if (!data.Title) {
      newErrors.Title = "Title is required.";
      isValid = false;
    }
    if (!data.Sentence) {
      newErrors.Sentence = "Sentence is required.";
      isValid = false;
    }
    if (!data.Question1) {
      newErrors.Question1 = "Question 1 is required.";
      isValid = false;
    }
    if (!data.Answer1) {
      newErrors.Answer1 = "Answer 1 is required.";
      isValid = false;
    }
    if (!data.Question2) {
      newErrors.Question2 = "Question 2 is required.";
      isValid = false;
    }
    if (!data.Answer2) {
      newErrors.Answer2 = "Answer 2 is required.";
      isValid = false;
    }
    if (!data.Question3) {
      newErrors.Question3 = "Question 3 is required.";
      isValid = false;
    }
    if (!data.Answer3) {
      newErrors.Answer3 = "Answer 3 is required.";
      isValid = false;
    }
    if (!data.Question4) {
      newErrors.Question4 = "Question 4 is required.";
      isValid = false;
    }
    if (!data.Answer4) {
      newErrors.Answer4 = "Answer 4 is required.";
      isValid = false;
    }
    if (!data.Question5) {
      newErrors.Question5 = "Question 5 is required.";
      isValid = false;
    }
    if (!data.Answer5) {
      newErrors.Answer5 = "Answer 5 is required.";
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const importSentence = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out all required fields.");
      return;
    }

    const {
      Type,
      Title,
      Sentence,
      Question1,
      Question2,
      Question3,
      Question4,
      Question5,
      Answer1,
      Answer2,
      Answer3,
      Answer4,
      Answer5,
    } = data;

    try {
      const response = await axios.post("/api/importSentence", {
        Type,
        Title,
        Sentence,
        Question1,
        Question2,
        Question3,
        Question4,
        Question5,
        Answer1,
        Answer2,
        Answer3,
        Answer4,
        Answer5,
      });

      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        toast.success("Sentence imported successfully.");
        setModalShow(true); // Show success modal
      }
    } catch (error) {
      console.error("Failed to import sentence:", error);
      toast.error("Failed to import sentence.");
    }
  };

  return (
    <>
      <Modal
        isOpen={show}
        onClose={onHide}
        size="2xl"
        aria-labelledby="import-modal-title"
        isDismissable={false}
        isKeyboardDismissDisabled={true}
        scrollBehavior="inside"
        placement="center"
      >
        <ModalContent className="w-full md:w-[50vw] max-w-full max-h-[80vh] overflow-y-auto bg-white p-4 rounded-lg">
          <ModalHeader id="import-modal-title">Import New Sentence</ModalHeader>
          <form onSubmit={importSentence}>
            <ModalBody>
              <div className="text-sm">
                <p>Please provide the information for your assessment.</p>
              </div>
              <Select
                labelPlacement="outside"
                label="Select type of Assessment"
                aria-label="Select activity type"
                defaultSelectedKeys={[" "]}
                value={data.Type}
                variant="bordered"
                className="bg-transparent py-1 my-1"
                onChange={(e) => setData({ ...data, Type: e.target.value })}
                isInvalid={!!errors.Type}
                errorMessage={errors.Type}
              >
                <SelectItem key=" ">Select Type of Assessment:</SelectItem>
                <SelectItem key="1">Assessment 4: Pagbabasa</SelectItem>
              </Select>

              <Input
                type="text"
                label="Title"
                placeholder="Enter a title"
                labelPlacement="outside"
                variant="bordered"
                className="bg-transparent py-1 my-1"
                value={data.Title}
                onChange={(e) => setData({ ...data, Title: e.target.value })}
                isInvalid={!!errors.Title}
                errorMessage={errors.Title}
                endContent={
                  <FontAwesomeIcon
                    icon={faBook}
                    className="text-2xl text-default-400 pointer-events-none flex-shrink-0"
                  />
                }
              />

              <Textarea
                placeholder="Input sentence"
                labelPlacement="outside"
                label="Add a Filipino short reading story"
                variant="bordered"
                className="bg-transparent py-1 my-1"
                value={data.Sentence}
                onChange={(e) => setData({ ...data, Sentence: e.target.value })}
                isInvalid={!!errors.Sentence}
                errorMessage={errors.Sentence}
                endContent={
                  <FontAwesomeIcon
                    icon={faPen}
                    className="text-2xl text-default-400 pointer-events-none flex-shrink-0"
                  />
                }
              />

              <div className="flex flex-col gap-6">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex flex-wrap gap-4">
                    <div className="flex flex-1">
                      <Input
                        type="text"
                        label={`Question ${i + 1}`}
                        placeholder="Write a question"
                        labelPlacement="outside"
                        variant="bordered"
                        className="bg-transparent py-1 my-1"
                        value={data[`Question${i + 1}`]}
                        onChange={(e) =>
                          setData({
                            ...data,
                            [`Question${i + 1}`]: e.target.value,
                          })
                        }
                        isInvalid={!!errors[`Question${i + 1}`]}
                        errorMessage={errors[`Question${i + 1}`]}
                        endContent={
                          <FontAwesomeIcon
                            icon={faQuestion}
                            className="text-2xl text-default-400 pointer-events-none flex-shrink-0"
                          />
                        }
                      />
                    </div>
                    <div className="flex flex-1">
                      <Input
                        type="text"
                        label={`Answer ${i + 1}`}
                        placeholder="Input the answer"
                        labelPlacement="outside"
                        variant="bordered"
                        className="bg-transparent py-1 my-1"
                        value={data[`Answer${i + 1}`]}
                        onChange={(e) =>
                          setData({
                            ...data,
                            [`Answer${i + 1}`]: e.target.value,
                          })
                        }
                        isInvalid={!!errors[`Answer${i + 1}`]}
                        errorMessage={errors[`Answer${i + 1}`]}
                        endContent={
                          <FontAwesomeIcon
                            icon={faComment}
                            className="text-2xl text-default-400 pointer-events-none flex-shrink-0"
                          />
                        }
                      />
                    </div>
                  </div>
                ))}
              </div>
            </ModalBody>
            <ModalFooter>
              <Button color="danger" variant="light" onClick={onHide}>
                Close
              </Button>
              <Button color="primary" type="submit">
                Import
              </Button>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
      <ImportSentenceSuccess
        show={modalShow}
        onHide={() => setModalShow(false)}
      />
    </>
  );
};

// Add prop types validation
ImportSentence.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
};

export default ImportSentence;
