import ParentSidebar from "../../Components/ParentDashboardComponent/ParentSidebar";
import BodyParentUpdate from "../../Components/ParentDashboardComponent/BodyParentUpdate";

const ProfileUpdate = () => {
  return (
    <div className="dashboard">
      <ParentSidebar />
      <div className="dashboard-content">
        <BodyParentUpdate />
      </div>
    </div>
  );
};

export default ProfileUpdate;
