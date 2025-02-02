import PropTypes from "prop-types"; // Import PropTypes
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";
import { Link } from "react-router-dom";

const TeacherViewPerformance = ({ show, onHide, performance }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col" id="edit-user-modal-title">
          View Performance Information
        </ModalHeader>
        <ModalBody>
          <p>Do you want to view the performance of the student selected?</p>
        </ModalBody>
        <ModalFooter>
          <Button color="danger" variant="light" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/ViewStudentPerformance/${performance?.UserInputId}`}>
            <Button color="primary">View</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

TeacherViewPerformance.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  performance: PropTypes.object,
};

export default TeacherViewPerformance;
